import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from pycocotools.coco import COCO
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_CLASSES = 80
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
LR = 0.001
K_PROBES = 28
EPSILON = 0.1
LAMBDA_JET = 0.1
ETA_JET = 0.5
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
MAX_TRAIN_BATCHES = int(os.getenv("MAX_TRAIN_BATCHES", "0"))

_cuda_env = os.getenv("CUDA_DEVICE")
if torch.cuda.is_available() and _cuda_env is not None:
    DEVICE = torch.device(f'cuda:{_cuda_env}')
else:
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("Using math SDPA backend for higher-order gradient compatibility.")


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


def _build_vit_backbone_coco():
    vit = models.vit_b_16(weights=None)
    return vit


def baseline_vit_b16(num_classes):
    model = _build_vit_backbone_coco()
    hidden_dim = model.heads.head.in_features
    model.heads.head = nn.Linear(hidden_dim, num_classes)
    return model


class COCOClassifierNFViT(nn.Module):
    def __init__(self, num_classes, k_probes, epsilon, n_fisher):
        super().__init__()
        if n_fisher not in (1, 2, 3):
            raise ValueError("n_fisher must be 1, 2, or 3")

        self.epsilon = epsilon
        self.n_fisher = n_fisher
        self.vit = _build_vit_backbone_coco()
        self.hidden_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()

        total_blocks = len(self.vit.encoder.layers)
        if n_fisher == 1:
            self.layer_groups = [list(range(total_blocks))]
        elif n_fisher == 2:
            split = total_blocks // 2
            self.layer_groups = [list(range(0, split)), list(range(split, total_blocks))]
        else:
            split1 = total_blocks // 3
            split2 = (2 * total_blocks) // 3
            self.layer_groups = [
                list(range(0, split1)),
                list(range(split1, split2)),
                list(range(split2, total_blocks)),
            ]

        v = torch.randn(k_probes, 3 * 224 * 224)
        v = v / torch.norm(v, dim=1, keepdim=True)
        self.register_buffer('probes', v)

        self.projections = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(n_fisher)])
        self.fc = nn.Linear(self.hidden_dim + n_fisher, num_classes)

    def _extract_selected_cls_features(self, x):
        x = self.vit._process_input(x)
        batch_size = x.shape[0]
        cls_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        layer_cls_features = []
        for idx, layer in enumerate(self.vit.encoder.layers):
            x = layer(x)
            layer_cls_features.append(self.vit.encoder.ln(x[:, 0]))

        selected = []
        for group in self.layer_groups:
            group_features = [layer_cls_features[i] for i in group]
            selected.append(torch.stack(group_features, dim=0).mean(dim=0))

        final_cls = self.vit.encoder.ln(x)[:, 0]
        return selected, final_cls

    def compute_score(self, x):
        if not x.requires_grad:
            x.requires_grad_(True)
        selected_features, _ = self._extract_selected_cls_features(x)
        energy = self.projections[0](selected_features[0]).sum()
        grads = torch.autograd.grad(
            outputs=energy,
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
            selected_features, final_feature = self._extract_selected_cls_features(x)
            fishers = []
            for feature, projection in zip(selected_features, self.projections):
                energy = projection(feature).sum()
                grads = torch.autograd.grad(
                    outputs=energy,
                    inputs=x,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                fishers.append(grads.flatten(1).pow(2).mean(dim=1, keepdim=True))

        logits = self.fc(torch.cat([final_feature, *fishers], dim=1))
        return logits, sum(fishers) / len(fishers)


class COCOClassifier1F(COCOClassifierNFViT):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__(num_classes=num_classes, k_probes=k_probes, epsilon=epsilon, n_fisher=1)


class COCOClassifier2F(COCOClassifierNFViT):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__(num_classes=num_classes, k_probes=k_probes, epsilon=epsilon, n_fisher=2)


class COCOClassifier3F(COCOClassifierNFViT):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__(num_classes=num_classes, k_probes=k_probes, epsilon=epsilon, n_fisher=3)


class COCOClassifierBetweenAllF(nn.Module):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.vit = _build_vit_backbone_coco()
        self.hidden_dim = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Identity()

        total_blocks = len(self.vit.encoder.layers)
        self.pair_count = total_blocks - 1

        v = torch.randn(k_probes, 3 * 224 * 224)
        v = v / torch.norm(v, dim=1, keepdim=True)
        self.register_buffer('probes', v)

        self.projections = nn.ModuleList([nn.Linear(self.hidden_dim, 1) for _ in range(self.pair_count)])
        self.fc = nn.Linear(self.hidden_dim + self.pair_count, num_classes)

    def _extract_between_features(self, x):
        x = self.vit._process_input(x)
        batch_size = x.shape[0]
        cls_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        layer_cls_features = []
        for layer in self.vit.encoder.layers:
            x = layer(x)
            layer_cls_features.append(self.vit.encoder.ln(x[:, 0]))

        between_features = [
            0.5 * (layer_cls_features[idx] + layer_cls_features[idx + 1])
            for idx in range(len(layer_cls_features) - 1)
        ]
        final_cls = self.vit.encoder.ln(x)[:, 0]
        return between_features, final_cls

    def compute_score(self, x):
        if not x.requires_grad:
            x.requires_grad_(True)
        between_features, _ = self._extract_between_features(x)
        energy = self.projections[0](between_features[0]).sum()
        grads = torch.autograd.grad(
            outputs=energy,
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
            between_features, final_feature = self._extract_between_features(x)
            fishers = []
            for feature, projection in zip(between_features, self.projections):
                energy = projection(feature).sum()
                grads = torch.autograd.grad(
                    outputs=energy,
                    inputs=x,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                fishers.append(grads.flatten(1).pow(2).mean(dim=1, keepdim=True))

        logits = self.fc(torch.cat([final_feature, *fishers], dim=1))
        return logits, sum(fishers) / len(fishers)


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
    train_losses, test_accuracies = [], []

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
        test_accuracies.append(acc)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Test Accuracy: {acc:.2f}%")

    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    return train_losses, test_accuracies


train_loader, val_loader = build_coco_loaders(BATCH_SIZE)

train_losses_baseline, test_accuracies_baseline = run_experiment(
    "Baseline ViT-B/16",
    baseline_vit_b16(NUM_CLASSES),
    train_loader,
    val_loader,
    use_jet=False,
    log_every=150,
)

train_losses_no_jet_1f, test_accuracies_no_jet_1f = run_experiment(
    "QI ViT-B/16 without Jet Loss",
    COCOClassifier1F(NUM_CLASSES, K_PROBES, EPSILON),
    train_loader,
    val_loader,
    use_jet=False,
)

train_losses_jet_1f, test_accuracies_jet_1f = run_experiment(
    "QI ViT-B/16 with Jet Loss",
    COCOClassifier1F(NUM_CLASSES, K_PROBES, EPSILON),
    train_loader,
    val_loader,
    use_jet=True,
)

train_losses_no_jet_2f, test_accuracies_no_jet_2f = run_experiment(
    "QI ViT-B/16 without Jet Loss 2F",
    COCOClassifier2F(NUM_CLASSES, K_PROBES, EPSILON),
    train_loader,
    val_loader,
    use_jet=False,
)

train_losses_no_jet_3f, test_accuracies_no_jet_3f = run_experiment(
    "QI ViT-B/16 without Jet Loss 3F",
    COCOClassifier3F(NUM_CLASSES, K_PROBES, EPSILON),
    train_loader,
    val_loader,
    use_jet=False,
)

train_losses_no_jet_between_allf, test_accuracies_no_jet_between_allf = run_experiment(
    "QI ViT-B/16 between every layer Fisher",
    COCOClassifierBetweenAllF(NUM_CLASSES, K_PROBES, EPSILON),
    train_loader,
    val_loader,
    use_jet=False,
)

print("\n--- Results Comparison ---")
epochs = list(range(EPOCHS))
results_df = pd.DataFrame({
    'Epoch': epochs,
    'Baseline_Train_Loss': train_losses_baseline,
    'Baseline_Test_Accuracy': test_accuracies_baseline,
    'No_Jet_Train_Loss_1F': train_losses_no_jet_1f,
    'No_Jet_Test_Accuracy_1F': test_accuracies_no_jet_1f,
    'Jet_Train_Loss_1F': train_losses_jet_1f,
    'Jet_Test_Accuracy_1F': test_accuracies_jet_1f,
    'No_Jet_Train_Loss_2F': train_losses_no_jet_2f,
    'No_Jet_Test_Accuracy_2F': test_accuracies_no_jet_2f,
    'No_Jet_Train_Loss_3F': train_losses_no_jet_3f,
    'No_Jet_Test_Accuracy_3F': test_accuracies_no_jet_3f,
    'No_Jet_Train_Loss_Between_AllF': train_losses_no_jet_between_allf,
    'No_Jet_Test_Accuracy_Between_AllF': test_accuracies_no_jet_between_allf,
})

csv_filename = 'training_results_comparison_vitb16_coco.csv'
results_df.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")
print("\nDataFrame Preview:")
print(results_df)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
for loss, label, marker, color in [
    (train_losses_baseline, 'Baseline ViT-B/16', 'o', 'blue'),
    (train_losses_no_jet_1f, 'QI no Jet 1F', 's', 'orange'),
    (train_losses_jet_1f, 'QI with Jet 1F', '^', 'red'),
    (train_losses_no_jet_2f, 'QI no Jet 2F', 'd', 'purple'),
    (train_losses_no_jet_3f, 'QI no Jet 3F', '<', 'brown'),
    (train_losses_no_jet_between_allf, 'QI between layers AllF', 'v', 'green'),
]:
    ax1.plot(epochs, loss, marker=marker, linewidth=2, markersize=6, label=label, color=color)

for acc, label, marker, color in [
    (test_accuracies_baseline, 'Baseline ViT-B/16', 'o', 'blue'),
    (test_accuracies_no_jet_1f, 'QI no Jet 1F', 's', 'orange'),
    (test_accuracies_jet_1f, 'QI with Jet 1F', '^', 'red'),
    (test_accuracies_no_jet_2f, 'QI no Jet 2F', 'd', 'purple'),
    (test_accuracies_no_jet_3f, 'QI no Jet 3F', '<', 'brown'),
    (test_accuracies_no_jet_between_allf, 'QI between layers AllF', 'v', 'green'),
]:
    ax2.plot(epochs, acc, marker=marker, linewidth=2, markersize=6, label=label, color=color)

ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs)

ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
ax2.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs)
ax2.set_ylim([90, 100])

plt.tight_layout()
plt.savefig('training_results_comparison_vitb16_coco.png', dpi=150, bbox_inches='tight')
print("\nComparison plot saved as 'training_results_comparison_vitb16_coco.png'")
plt.show()

print("\n" + "=" * 60)
print("Final Test Accuracy Summary:")
print("=" * 60)
print(f"Baseline ViT-B/16:          {test_accuracies_baseline[-1]:.2f}%")
print(f"QI without Jet Loss 1F:     {test_accuracies_no_jet_1f[-1]:.2f}%")
print(f"QI with Jet Loss 1F:        {test_accuracies_jet_1f[-1]:.2f}%")
print(f"QI without Jet Loss 2F:     {test_accuracies_no_jet_2f[-1]:.2f}%")
print(f"QI without Jet Loss 3F:     {test_accuracies_no_jet_3f[-1]:.2f}%")
print(f"QI between layers AllF:     {test_accuracies_no_jet_between_allf[-1]:.2f}%")
print("=" * 60)
