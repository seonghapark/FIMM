import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms, models

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

NUM_CLASSES = 10
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
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
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


def build_loaders(batch_size, num_workers=NUM_WORKERS):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    return train_loader, test_loader


def baseline_resnet101(num_classes):
    model = models.resnet101(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class MNISTClassifier1F(nn.Module):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.k_probes = k_probes

        v = torch.randn(k_probes, 28 * 28)
        v = v / torch.norm(v, dim=1, keepdim=True)
        self.register_buffer('probes', v)

        resnet_model = models.resnet101(weights=None)
        resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model_upper = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
            resnet_model.layer1,
            resnet_model.layer2,
        )
        self.scalar_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),
        )
        self.model_lower = nn.Sequential(
            nn.Conv2d(131, 128, kernel_size=1, padding=0, bias=True),
            resnet_model.layer3,
            resnet_model.layer4,
            resnet_model.avgpool,
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def compute_score(self, x):
        if not x.requires_grad:
            x.requires_grad_(True)
        h = self.model_upper(x)
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
            mean_v = self.probes.mean(dim=0, keepdim=True).view(1, 1, 28, 28)
            i_pos = self.compute_local_fisher(x + mean_v)
            i_neg = self.compute_local_fisher(x - mean_v)
            o1 = (i_pos - i_neg) / (2 * self.epsilon)
            o2 = (i_pos - 2 * o0 + i_neg) / (self.epsilon ** 2)

        h = self.model_upper(x)
        size = (h.size(2), h.size(3))
        o0b = o0.view(-1, 1, 1, 1).expand(-1, 1, *size)
        o1b = o1.view(-1, 1, 1, 1).expand(-1, 1, *size)
        o2b = o2.view(-1, 1, 1, 1).expand(-1, 1, *size)
        logits = self.model_lower(torch.cat([h, o0b, o1b, o2b], dim=1))
        return logits, o0


class MNISTClassifier2F(nn.Module):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__()

        resnet_model = models.resnet101(weights=None)
        resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

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

        self.proj1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 1))
        self.proj2 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, 1))
        self.fc = nn.Linear(512 + 2, num_classes)

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
            f2 = self._feature_fisher(h2, self.proj2)

        h = self.layer4(self.layer3(h2))
        pooled = self.avgpool(h).flatten(1)
        logits = self.fc(torch.cat([pooled, f1, f2], dim=1))
        return logits, (f1 + f2) / 2


class MNISTClassifier3F(nn.Module):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__()

        resnet_model = models.resnet101(weights=None)
        resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

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

        self.proj1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 1))
        self.proj2 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, 1))
        self.proj3 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 1))
        self.fc = nn.Linear(512 + 3, num_classes)

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
            f1 = self._feature_fisher(h1, self.proj1)
            f2 = self._feature_fisher(h2, self.proj2)
            f3 = self._feature_fisher(h3, self.proj3)

        h = self.layer4(h3)
        pooled = self.avgpool(h).flatten(1)
        logits = self.fc(torch.cat([pooled, f1, f2, f3], dim=1))
        return logits, (f1 + f2 + f3) / 3


class MNISTClassifierAllF(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnet101(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

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

        self.feature_names = [
            "stem",
            "layer1_block0",
            "layer1_block1",
            "layer2_block0",
            "layer2_block1",
            "layer3_block0",
            "layer3_block1",
            "layer4_block0",
            "layer4_block1",
        ]

        channel_map = {
            "stem": 64,
            "layer1_block0": 64,
            "layer1_block1": 64,
            "layer2_block0": 128,
            "layer2_block1": 128,
            "layer3_block0": 256,
            "layer3_block1": 256,
            "layer4_block0": 512,
            "layer4_block1": 512,
        }

        self.fisher_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(channel_map[name], 1),
            )
            for name in self.feature_names
        })

        self.classifier = nn.Linear(512 + len(self.feature_names), num_classes)

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

        h = self.layer1[0](h)
        features["layer1_block0"] = h
        h = self.layer1[1](h)
        features["layer1_block1"] = h

        h = self.layer2[0](h)
        features["layer2_block0"] = h
        h = self.layer2[1](h)
        features["layer2_block1"] = h

        h = self.layer3[0](h)
        features["layer3_block0"] = h
        h = self.layer3[1](h)
        features["layer3_block1"] = h

        h = self.layer4[0](h)
        features["layer4_block0"] = h
        h = self.layer4[1](h)
        features["layer4_block1"] = h

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


def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            logits = unpack_logits(model(data))
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


def jet_regularizer(model, data):
    mean_v = model.probes.mean(dim=0, keepdim=True).view(1, 1, 28, 28)
    pred_pos, _ = model(data + EPSILON * mean_v)
    pred_neg, _ = model(data - EPSILON * mean_v)
    d_hat = (pred_pos - pred_neg) / (2 * EPSILON)
    score_proj = model.compute_score(data).mean(dim=1, keepdim=True)
    return ((d_hat + LAMBDA_JET * score_proj) ** 2).mean()


def run_experiment(name, model, train_loader, test_loader, use_jet=False, log_every=50):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_losses, test_accuracies = [], []

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
        acc = evaluate(model, test_loader)
        train_losses.append(avg_loss)
        test_accuracies.append(acc)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Test Accuracy: {acc:.2f}%")

    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    return train_losses, test_accuracies


if __name__ == "__main__":
    train_loader, test_loader = build_loaders(BATCH_SIZE, NUM_WORKERS)

    train_losses_baseline, test_accuracies_baseline = run_experiment(
        "Baseline ResNet101",
        baseline_resnet101(NUM_CLASSES),
        train_loader,
        test_loader,
        use_jet=False,
        log_every=150,
    )

    train_losses_no_jet_1f, test_accuracies_no_jet_1f = run_experiment(
        "QI ResNet101 without Jet Loss 1F",
        MNISTClassifier1F(NUM_CLASSES, K_PROBES, EPSILON),
        train_loader,
        test_loader,
        use_jet=False,
    )

    train_losses_jet_1f, test_accuracies_jet_1f = run_experiment(
        "QI ResNet101 with Jet Loss 1F",
        MNISTClassifier1F(NUM_CLASSES, K_PROBES, EPSILON),
        train_loader,
        test_loader,
        use_jet=True,
    )

    train_losses_no_jet_2f, test_accuracies_no_jet_2f = run_experiment(
        "QI ResNet101 without Jet Loss 2F",
        MNISTClassifier2F(NUM_CLASSES, K_PROBES, EPSILON),
        train_loader,
        test_loader,
        use_jet=False,
    )

    train_losses_jet_2f, test_accuracies_jet_2f = run_experiment(
        "QI ResNet101 with Jet Loss 2F",
        MNISTClassifier2F(NUM_CLASSES, K_PROBES, EPSILON),
        train_loader,
        test_loader,
        use_jet=True,
    )

    train_losses_no_jet_3f, test_accuracies_no_jet_3f = run_experiment(
        "QI ResNet101 without Jet Loss 3F",
        MNISTClassifier3F(NUM_CLASSES, K_PROBES, EPSILON),
        train_loader,
        test_loader,
        use_jet=False,
    )

    train_losses_jet_3f, test_accuracies_jet_3f = run_experiment(
        "QI ResNet101 with Jet Loss 3F",
        MNISTClassifier3F(NUM_CLASSES, K_PROBES, EPSILON),
        train_loader,
        test_loader,
        use_jet=True,
    )

    train_losses_no_jet_allf, test_accuracies_no_jet_allf = run_experiment(
        "QI ResNet101 with Fisher from Every Layer",
        MNISTClassifierAllF(NUM_CLASSES),
        train_loader,
        test_loader,
        use_jet=False,
        log_every=150,
    )

    train_losses_jet_allf, test_accuracies_jet_allf = run_experiment(
        "QI ResNet101 with Fisher from Every Layer",
        MNISTClassifierAllF(NUM_CLASSES),
        train_loader,
        test_loader,
        use_jet=True,
        log_every=150,
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
        'Jet_Train_Loss_2F': train_losses_jet_2f,
        'Jet_Test_Accuracy_2F': test_accuracies_jet_2f,
        'No_Jet_Train_Loss_3F': train_losses_no_jet_3f,
        'No_Jet_Test_Accuracy_3F': test_accuracies_no_jet_3f,
        'Jet_Train_Loss_3F': train_losses_jet_3f,
        'Jet_Test_Accuracy_3F': test_accuracies_jet_3f,
        'No_Jet_AllF_Train_Loss': train_losses_no_jet_allf,
        'No_Jet_AllF_Test_Accuracy': test_accuracies_no_jet_allf,
        'Jet_AllF_Train_Loss': train_losses_jet_allf,
        'Jet_AllF_Test_Accuracy': test_accuracies_jet_allf,
    })

    csv_filename = 'training_results_mnist_all_final.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
    print("\nDataFrame Preview:")
    print(results_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for loss, label, marker, color in [
        (train_losses_baseline, 'Baseline ResNet101', 'o', 'blue'),
        (train_losses_no_jet_1f, 'QI no Jet 1F', 's', 'orange'),
        (train_losses_jet_1f, 'QI with Jet 1F', '^', 'red'),
        (train_losses_no_jet_2f, 'QI no Jet 2F', 'd', 'purple'),
        (train_losses_jet_2f, 'QI with Jet 2F', 'v', 'green'),
        (train_losses_no_jet_3f, 'QI no Jet 3F', '<', 'brown'),
        (train_losses_jet_3f, 'QI with Jet 3F', '>', 'pink'),
        (train_losses_no_jet_allf, 'QI no Jet AllF', 'x', 'black'),
        (train_losses_jet_allf, 'QI with Jet AllF', '+', 'cyan'),
    ]:
        ax1.plot(epochs, loss, marker=marker, linewidth=2, markersize=6, label=label, color=color)

    for acc, label, marker, color in [
        (test_accuracies_baseline, 'Baseline ResNet101', 'o', 'blue'),
        (test_accuracies_no_jet_1f, 'QI no Jet 1F', 's', 'orange'),
        (test_accuracies_jet_1f, 'QI with Jet 1F', '^', 'red'),
        (test_accuracies_no_jet_2f, 'QI no Jet 2F', 'd', 'purple'),
        (test_accuracies_jet_2f, 'QI with Jet 2F', 'v', 'green'),
        (test_accuracies_no_jet_3f, 'QI no Jet 3F', '<', 'brown'),
        (test_accuracies_jet_3f, 'QI with Jet 3F', '>', 'pink'),
        (test_accuracies_no_jet_allf, 'QI no Jet AllF', 'x', 'black'),
        (test_accuracies_jet_allf, 'QI with Jet AllF', '+', 'cyan'),
    ]:
        ax2.plot(epochs, acc, marker=marker, linewidth=2, markersize=6, label=label, color=color)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('MNIST Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('MNIST Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim([90, 100])

    plt.tight_layout()
    plt.savefig('training_results_mnist_all_final.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'training_results_mnist_all_final.png'")
    plt.show()

    print("\n" + "=" * 60)
    print("Final Test Accuracy Summary:")
    print("=" * 60)
    print(f"Baseline ResNet101:          {test_accuracies_baseline[-1]:.2f}%")
    print(f"QI without Jet Loss 1F:     {test_accuracies_no_jet_1f[-1]:.2f}%")
    print(f"QI with Jet Loss 1F:        {test_accuracies_jet_1f[-1]:.2f}%")
    print(f"QI without Jet Loss 2F:     {test_accuracies_no_jet_2f[-1]:.2f}%")
    print(f"QI with Jet Loss 2F:        {test_accuracies_jet_2f[-1]:.2f}%")
    print(f"QI without Jet Loss 3F:     {test_accuracies_no_jet_3f[-1]:.2f}%")
    print(f"QI with Jet Loss 3F:        {test_accuracies_jet_3f[-1]:.2f}%")
    print(f"QI without Jet Loss AllF:   {test_accuracies_no_jet_allf[-1]:.2f}%")
    print(f"QI with Jet Loss AllF:      {test_accuracies_jet_allf[-1]:.2f}%")
    print("=" * 60)
