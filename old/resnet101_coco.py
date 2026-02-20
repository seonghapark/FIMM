import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
LR = 0.001
NUM_CLASSES = 80
K_PROBES = 224
EPSILON = 0.1
LAMBDA_JET = 0.1
ETA_JET = 0.5
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "10"))
MAX_TRAIN_BATCHES = int(os.getenv("MAX_TRAIN_BATCHES", "0"))
MAX_EXPERIMENTS = int(os.getenv("MAX_EXPERIMENTS", "0"))

_cuda_env = os.getenv("CUDA_DEVICE")
if torch.cuda.is_available() and _cuda_env is not None:
    DEVICE = torch.device(f'cuda:{_cuda_env}')
else:
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class COCOSingleLabelDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_root, category_to_index=None, image_size=224):
        self.coco = COCO(annotation_file)
        self.image_root = image_root
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        category_ids = sorted(self.coco.getCatIds())
        self.category_to_index = category_to_index or {cat_id: idx for idx, cat_id in enumerate(category_ids)}

        self.samples = []
        for image_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
            if not ann_ids:
                continue

            annotations = self.coco.loadAnns(ann_ids)
            category_id = next((ann['category_id'] for ann in annotations if ann['category_id'] in self.category_to_index), None)
            if category_id is None:
                continue

            image_info = self.coco.loadImgs(image_id)[0]
            image_path = os.path.join(self.image_root, image_info['file_name'])
            if os.path.exists(image_path):
                self.samples.append((image_path, self.category_to_index[category_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), label


def build_coco_loaders(batch_size, data_root='./data/coco', image_size=224, num_workers=NUM_WORKERS):
    train_ann = os.path.join(data_root, 'annotations', 'instances_train2017.json')
    val_ann = os.path.join(data_root, 'annotations', 'instances_val2017.json')
    train_img_root = os.path.join(data_root, 'train2017')
    val_img_root = os.path.join(data_root, 'val2017')

    if not all(os.path.exists(path) for path in [train_ann, val_ann, train_img_root, val_img_root]):
        raise FileNotFoundError(
            f"COCO dataset paths not found under {data_root}. Expected annotations and train2017/val2017 folders."
        )

    print("Loading COCO dataset...")
    train_dataset = COCOSingleLabelDataset(train_ann, train_img_root, image_size=image_size)
    val_dataset = COCOSingleLabelDataset(
        val_ann,
        val_img_root,
        category_to_index=train_dataset.category_to_index,
        image_size=image_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Number of classes in mapping: {len(train_dataset.category_to_index)}")
    return train_loader, val_loader


def build_resnet101_backbone():
    model = models.resnet101(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def baseline_resnet101(num_classes):
    model = build_resnet101_backbone()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class COCOClassifier1F(nn.Module):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.k_probes = k_probes

        v = torch.randn(k_probes, 3 * 224 * 224)
        v = v / torch.norm(v, dim=1, keepdim=True)
        self.register_buffer('probes', v)

        resnet_model = build_resnet101_backbone()

        self.stem = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
        )
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool
        self.scalar_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1),
        )
        self.classifier = nn.Linear(2048 + 3, num_classes)

    def _extract_final_map(self, x):
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        return h

    def compute_score(self, x):
        if not x.requires_grad:
            x.requires_grad_(True)
        h = self._extract_final_map(x)
        energy = self.scalar_projection(h)
        grads = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
        )[0]
        return -torch.matmul(grads.flatten(1), self.probes.T)

    def compute_local_fisher(self, x):
        return (self.compute_score(x) ** 2).mean(dim=1, keepdim=True)

    def forward(self, x):
        with torch.set_grad_enabled(True):
            if not x.requires_grad:
                x.requires_grad_(True)
            o0 = self.compute_local_fisher(x)
            mean_v = self.probes.mean(dim=0, keepdim=True).view(1, 3, 224, 224)
            i_pos = self.compute_local_fisher(x + mean_v)
            i_neg = self.compute_local_fisher(x - mean_v)
            o1 = (i_pos - i_neg) / (2 * self.epsilon)
            o2 = (i_pos - 2 * o0 + i_neg) / (self.epsilon ** 2)

        h = self._extract_final_map(x)
        pooled = self.avgpool(h).flatten(1)
        logits = self.classifier(torch.cat([pooled, o0, o1, o2], dim=1))
        return logits, o0


class COCOClassifier2F(nn.Module):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__()

        resnet_model = build_resnet101_backbone()

        self.stem1 = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
            resnet_model.layer1,
        )
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool

        self.proj1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1))
        self.proj2 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(2048, 1))
        self.fc = nn.Linear(2048 + 2, num_classes)

    @staticmethod
    def _feature_fisher(feature_map, projection):
        if not feature_map.requires_grad:
            feature_map.requires_grad_(True)
        energy = projection(feature_map).sum()
        grads = torch.autograd.grad(energy, feature_map, create_graph=True, retain_graph=True)[0]
        return grads.flatten(1).pow(2).mean(dim=1, keepdim=True)

    def forward(self, x):
        with torch.set_grad_enabled(True):
            h1 = self.stem1(x)
            h2 = self.layer2(h1)
            f1 = self._feature_fisher(h1, self.proj1)
            h3 = self.layer3(h2)
            h = self.layer4(h3)
            f2 = self._feature_fisher(h, self.proj2)

        pooled = self.avgpool(h).flatten(1)
        logits = self.fc(torch.cat([pooled, f1, f2], dim=1))
        return logits, (f1 + f2) / 2


class COCOClassifier3F(nn.Module):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__()

        resnet_model = build_resnet101_backbone()

        self.stem1 = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
            resnet_model.layer1,
        )
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool

        self.proj1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1))
        self.proj2 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 1))
        self.proj3 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(2048, 1))
        self.fc = nn.Linear(2048 + 3, num_classes)

    @staticmethod
    def _feature_fisher(feature_map, projection):
        if not feature_map.requires_grad:
            feature_map.requires_grad_(True)
        energy = projection(feature_map).sum()
        grads = torch.autograd.grad(energy, feature_map, create_graph=True, retain_graph=True)[0]
        return grads.flatten(1).pow(2).mean(dim=1, keepdim=True)

    def forward(self, x):
        with torch.set_grad_enabled(True):
            h1 = self.stem1(x)
            h2 = self.layer2(h1)
            h3 = self.layer3(h2)
            h4 = self.layer4(h3)
            f1 = self._feature_fisher(h1, self.proj1)
            f2 = self._feature_fisher(h2, self.proj2)
            f3 = self._feature_fisher(h4, self.proj3)

        pooled = self.avgpool(h4).flatten(1)
        logits = self.fc(torch.cat([pooled, f1, f2, f3], dim=1))
        return logits, (f1 + f2 + f3) / 3


class COCOClassifierAllF(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = build_resnet101_backbone()

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        feature_specs = [("stem", 64)]
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(self, layer_name)
            for block_idx, block in enumerate(layer):
                out_channels = block.conv3.out_channels if hasattr(block, 'conv3') else block.conv2.out_channels
                feature_specs.append((f"{layer_name}_block{block_idx}", out_channels))

        self.feature_names = [name for name, _ in feature_specs]

        self.fisher_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(channels, 1),
            )
            for name, channels in feature_specs
        })

        self.classifier = nn.Linear(2048 + len(self.feature_names), num_classes)

    @staticmethod
    def _fisher_term(feature_map, projection_head):
        energy = projection_head(feature_map).sum()
        grads = torch.autograd.grad(
            outputs=energy,
            inputs=feature_map,
            create_graph=True,
            retain_graph=True,
        )[0]
        return grads.flatten(1).pow(2).mean(dim=1, keepdim=True)

    def _extract_backbone_features(self, x):
        features = {}

        h = self.stem(x)
        features["stem"] = h

        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            layer = getattr(self, layer_name)
            for block_idx, block in enumerate(layer):
                h = block(h)
                features[f"{layer_name}_block{block_idx}"] = h

        return features, h

    def forward(self, x):
        with torch.set_grad_enabled(True):
            if not x.requires_grad:
                x.requires_grad_(True)

            feature_maps, final_map = self._extract_backbone_features(x)
            fisher_terms = [
                self._fisher_term(feature_maps[name], self.fisher_heads[name])
                for name in self.feature_names
            ]

        pooled = self.avgpool(final_map).flatten(1)
        logits = self.classifier(torch.cat([pooled, *fisher_terms], dim=1))
        fisher_matrix = torch.cat(fisher_terms, dim=1)
        return logits, fisher_matrix


def unpack_logits(output):
    return output[0] if isinstance(output, tuple) else output


def evaluate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            logits = unpack_logits(model(data))
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


def jet_regularizer(model, data):
    mean_v = model.probes.mean(dim=0, keepdim=True).view(1, 3, 224, 224)
    pred_pos, _ = model(data + EPSILON * mean_v)
    pred_neg, _ = model(data - EPSILON * mean_v)
    d_hat = (pred_pos - pred_neg) / (2 * EPSILON)
    score_proj = model.compute_score(data).mean(dim=1, keepdim=True)
    return ((d_hat + LAMBDA_JET * score_proj) ** 2).mean()


def run_experiment(name, model, train_loader, val_loader, use_jet=False, log_every=50):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_losses, val_accuracies = [], []

    jet_supported = hasattr(model, 'compute_score') and hasattr(model, 'probes')
    if use_jet and not jet_supported:
        print(f"[Info] Jet loss is not supported for {model.__class__.__name__}; running without Jet loss.")
        use_jet = False

    print(f"\n--- {name} ---")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            logits = unpack_logits(model(data))
            loss = criterion(logits, target)
            if use_jet:
                loss = loss + ETA_JET * jet_regularizer(model, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % log_every == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")

            if MAX_TRAIN_BATCHES > 0 and (batch_idx + 1) >= MAX_TRAIN_BATCHES:
                break

        avg_loss = epoch_loss / len(train_loader)
        acc = evaluate(model, val_loader)
        train_losses.append(avg_loss)
        val_accuracies.append(acc)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Accuracy: {acc:.2f}%")

    print(f"Final Val Accuracy: {val_accuracies[-1]:.2f}%")
    return train_losses, val_accuracies


if __name__ == '__main__':
    train_loader, val_loader = build_coco_loaders(BATCH_SIZE, image_size=224)

    experiments = [
        {
            'key': 'baseline',
            'name': 'Baseline ResNet101',
            'build': lambda: baseline_resnet101(NUM_CLASSES),
            'use_jet': False,
            'log_every': 150,
            'loss_col': 'Baseline_Train_Loss',
            'acc_col': 'Baseline_Val_Accuracy',
            'marker': 'o',
            'color': 'blue',
        },
        {
            'key': 'no_jet_1f',
            'name': 'QI ResNet101 without Jet Loss 1F',
            'build': lambda: COCOClassifier1F(NUM_CLASSES, K_PROBES, EPSILON),
            'use_jet': False,
            'log_every': 50,
            'loss_col': 'No_Jet_Train_Loss_1F',
            'acc_col': 'No_Jet_Val_Accuracy_1F',
            'marker': 's',
            'color': 'orange',
        },
        {
            'key': 'jet_1f',
            'name': 'QI ResNet101 with Jet Loss 1F',
            'build': lambda: COCOClassifier1F(NUM_CLASSES, K_PROBES, EPSILON),
            'use_jet': True,
            'log_every': 50,
            'loss_col': 'Jet_Train_Loss_1F',
            'acc_col': 'Jet_Val_Accuracy_1F',
            'marker': '^',
            'color': 'red',
        },
        {
            'key': 'no_jet_2f',
            'name': 'QI ResNet101 without Jet Loss 2F',
            'build': lambda: COCOClassifier2F(NUM_CLASSES, K_PROBES, EPSILON),
            'use_jet': False,
            'log_every': 50,
            'loss_col': 'No_Jet_Train_Loss_2F',
            'acc_col': 'No_Jet_Val_Accuracy_2F',
            'marker': 'd',
            'color': 'purple',
        },
        {
            'key': 'jet_2f',
            'name': 'QI ResNet101 with Jet Loss 2F',
            'build': lambda: COCOClassifier2F(NUM_CLASSES, K_PROBES, EPSILON),
            'use_jet': True,
            'log_every': 50,
            'loss_col': 'Jet_Train_Loss_2F',
            'acc_col': 'Jet_Val_Accuracy_2F',
            'marker': 'v',
            'color': 'green',
        },
        {
            'key': 'no_jet_3f',
            'name': 'QI ResNet101 without Jet Loss 3F',
            'build': lambda: COCOClassifier3F(NUM_CLASSES, K_PROBES, EPSILON),
            'use_jet': False,
            'log_every': 50,
            'loss_col': 'No_Jet_Train_Loss_3F',
            'acc_col': 'No_Jet_Val_Accuracy_3F',
            'marker': '<',
            'color': 'brown',
        },
        {
            'key': 'jet_3f',
            'name': 'QI ResNet101 with Jet Loss 3F',
            'build': lambda: COCOClassifier3F(NUM_CLASSES, K_PROBES, EPSILON),
            'use_jet': True,
            'log_every': 50,
            'loss_col': 'Jet_Train_Loss_3F',
            'acc_col': 'Jet_Val_Accuracy_3F',
            'marker': '>',
            'color': 'pink',
        },
        {
            'key': 'no_jet_allf',
            'name': 'ResNet101 without Jet Loss and Fisher from Every Layer',
            'build': lambda: COCOClassifierAllF(NUM_CLASSES),
            'use_jet': False,
            'log_every': 100,
            'loss_col': 'No_Jet_Train_Loss_AllF',
            'acc_col': 'No_Jet_Val_Accuracy_AllF',
            'marker': 'x',
            'color': 'black',
        },
        {
            'key': 'jet_allf',
            'name': 'ResNet101 with Jet Loss and Fisher from Every Layer',
            'build': lambda: COCOClassifierAllF(NUM_CLASSES),
            'use_jet': True,
            'log_every': 100,
            'loss_col': 'Jet_Train_Loss_AllF',
            'acc_col': 'Jet_Val_Accuracy_AllF',
            'marker': '+',
            'color': 'cyan',
        },
    ]

    if MAX_EXPERIMENTS > 0:
        experiments = experiments[:MAX_EXPERIMENTS]

    results = {}
    for spec in experiments:
        results[spec['key']] = run_experiment(
            spec['name'],
            spec['build'](),
            train_loader,
            val_loader,
            use_jet=spec['use_jet'],
            log_every=spec['log_every'],
        )

    print("\n--- ResNet101 Results Comparison ---")
    epochs = list(range(EPOCHS))
    results_df = pd.DataFrame({'Epoch': epochs})
    for spec in experiments:
        losses, accs = results[spec['key']]
        results_df[spec['loss_col']] = losses
        results_df[spec['acc_col']] = accs

    csv_filename = 'training_results_resnet101_coco.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    print("\nDataFrame Preview:")
    print(results_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for spec in experiments:
        losses, accs = results[spec['key']]
        ax1.plot(
            epochs,
            losses,
            marker=spec['marker'],
            linewidth=2,
            markersize=6,
            label=spec['name'],
            color=spec['color'],
        )
        ax2.plot(
            epochs,
            accs,
            marker=spec['marker'],
            linewidth=2,
            markersize=6,
            label=spec['name'],
            color=spec['color'],
        )

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('ResNet101 COCO Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Val Accuracy (%)', fontsize=12)
    ax2.set_title('ResNet101 COCO Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig('training_results_resnet101_coco.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'training_results_resnet101_coco.png'")
    plt.show()

    print("\n" + "=" * 60)
    print("Final Validation Accuracy Summary:")
    print("=" * 60)
    for spec in experiments:
        _, accs = results[spec['key']]
        print(f"{spec['name']:56} {accs[-1]:.2f}%")
    print("=" * 60)
