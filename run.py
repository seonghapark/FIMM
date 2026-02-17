import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import torch.nn.functional as F
from pycocotools.coco import COCO
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
NUM_CLASSES = 80
K_PROBES = 224
EPSILON = 0.1
LAMBDA_JET = 0.1
ETA_JET = 0.5

# Device setup
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Custom COCO Dataset for Classification
class COCOClassification(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.transform = transform

        # Get all image ids that have annotations
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Filter out images without annotations
        self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]

        # Create mapping from COCO category IDs to continuous labels [0, 79]
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Get the primary category (most common or first)
        if len(anns) > 0:
            cat_id = anns[0]['category_id']
            label = self.cat_id_to_label[cat_id]
        else:
            label = 0

        # Load image
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.ids)


# QI Classifier with Fisher Information
class COCOClassifier(nn.Module):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.k_probes = k_probes
        self.num_classes = num_classes

        v = torch.randn(k_probes, 224*224*3)
        v = v / torch.norm(v, dim=1, keepdim=True)
        self.register_buffer('probes', v)

        # ResNet18 backbone (RGB input)
        resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)

        self.model_upper = nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu,
            resnet_model.maxpool,
            resnet_model.layer1,
            resnet_model.layer2,
        )

        self.scalar_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=1, bias=True)
        )

        self.model_lower = nn.Sequential(
            nn.Conv2d(128 + 3, 128, kernel_size=1, padding=0, bias=True),
            resnet_model.layer3,
            resnet_model.layer4,
            resnet_model.avgpool,
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
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
            retain_graph=True
        )[0]

        grads_flat = grads.view(grads.size(0), -1)
        probes_flat = self.probes.view(self.k_probes, -1)

        scores = -torch.matmul(grads_flat, probes_flat.T)
        return scores

    def compute_local_fisher(self, x):
        scores = self.compute_score(x)
        fisher_info = (scores ** 2).mean(dim=1, keepdim=True)
        return fisher_info

    def forward(self, x):
        with torch.set_grad_enabled(True):
            if not x.requires_grad:
                x.requires_grad_(True)

            O0 = self.compute_local_fisher(x)

            mean_v = self.probes.mean(dim=0, keepdim=True).view(1, 3, 224, 224)
            x_pos = x + mean_v
            x_neg = x - mean_v

            I_pos = self.compute_local_fisher(x_pos)
            I_neg = self.compute_local_fisher(x_neg)

            O1 = (I_pos - I_neg) / (2 * self.epsilon)
            O2 = (I_pos - 2 * O0 + I_neg) / (self.epsilon ** 2)

        h = self.model_upper(x)

        O0_broadcast = O0.view(-1, 1, 1, 1).expand(-1, 1, h.size(2), h.size(3))
        O1_broadcast = O1.view(-1, 1, 1, 1).expand(-1, 1, h.size(2), h.size(3))
        O2_broadcast = O2.view(-1, 1, 1, 1).expand(-1, 1, h.size(2), h.size(3))

        features = torch.cat([h, O0_broadcast, O1_broadcast, O2_broadcast], dim=1)
        y_hat = self.model_lower(features)

        return y_hat, O0


def load_coco_data():
    """Load COCO dataset and create data loaders."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading COCO dataset...")
    train_dataset = COCOClassification(
        root='./data/coco/train2017',
        annFile='./data/coco/annotations/instances_train2017.json',
        transform=transform
    )
    test_dataset = COCOClassification(
        root='./data/coco/val2017',
        annFile='./data/coco/annotations/instances_val2017.json',
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    return train_loader, test_loader


def train_baseline_resnet(train_loader, test_loader):
    """Train baseline ResNet18 model."""
    print("\n" + "="*60)
    print("Training Baseline ResNet18")
    print("="*60)
    
    resnet18 = models.resnet18(pretrained=False)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = resnet18.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 500 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")
        
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    return train_losses, test_accuracies


def train_qi_model(train_loader, test_loader, use_jet_loss=False):
    """Train QI model with or without Jet Loss."""
    model_name = "QI with Jet Loss" if use_jet_loss else "QI without Jet Loss"
    print("\n" + "="*60)
    print(f"Training {model_name}")
    print("="*60)
    
    model = COCOClassifier(NUM_CLASSES, K_PROBES, EPSILON)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output, energy_score = model(data)
            loss = criterion(output, target)

            if use_jet_loss:
                mean_v = model.probes.mean(dim=0, keepdim=True).view(1, 3, 224, 224)
                x_pos = data + EPSILON * mean_v
                x_neg = data - EPSILON * mean_v
                pred_pos, _ = model(x_pos)
                pred_neg, _ = model(x_neg)
                D_hat = (pred_pos - pred_neg) / (2 * EPSILON)
                score_proj = model.compute_score(data).mean(dim=1, keepdim=True)
                loss_jet = ((D_hat + LAMBDA_JET * score_proj)**2).mean()
                loss += ETA_JET * loss_jet

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 500 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, energy_score = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    return train_losses, test_accuracies


def save_and_plot_results(train_losses_baseline, test_accuracies_baseline,
                          train_losses_no_jet, test_accuracies_no_jet,
                          train_losses_jet, test_accuracies_jet):
    """Save results to CSV and create comparison plots."""
    print("\n" + "="*60)
    print("Saving Results and Creating Plots")
    print("="*60)
    
    epochs = list(range(EPOCHS))
    results_df = pd.DataFrame({
        'Epoch': epochs,
        'Baseline_Train_Loss': train_losses_baseline,
        'Baseline_Test_Accuracy': test_accuracies_baseline,
        'No_Jet_Train_Loss': train_losses_no_jet,
        'No_Jet_Test_Accuracy': test_accuracies_no_jet,
        'Jet_Train_Loss': train_losses_jet,
        'Jet_Test_Accuracy': test_accuracies_jet
    })

    # Save to CSV
    csv_filename = 'training_results_comparison.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")

    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Training Loss Comparison
    ax1.plot(epochs, train_losses_baseline, marker='o', linewidth=2, markersize=6, 
             label='Baseline ResNet18', color='blue')
    ax1.plot(epochs, train_losses_no_jet, marker='s', linewidth=2, markersize=6, 
             label='QI without Jet Loss', color='orange')
    ax1.plot(epochs, train_losses_jet, marker='^', linewidth=2, markersize=6, 
             label='QI with Jet Loss', color='red')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Comparison (COCO)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # Plot 2: Test Accuracy Comparison
    ax2.plot(epochs, test_accuracies_baseline, marker='o', linewidth=2, markersize=6, 
             label='Baseline ResNet18', color='blue')
    ax2.plot(epochs, test_accuracies_no_jet, marker='s', linewidth=2, markersize=6, 
             label='QI without Jet Loss', color='orange')
    ax2.plot(epochs, test_accuracies_jet, marker='^', linewidth=2, markersize=6, 
             label='QI with Jet Loss', color='red')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy Comparison (COCO)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    plt.tight_layout()
    plt.savefig('training_results_comparison_coco.png', dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved as 'training_results_comparison_coco.png'")
    plt.show()

    # Print final accuracies
    print("\n" + "="*60)
    print("Final Test Accuracy Summary:")
    print("="*60)
    print(f"Baseline ResNet18:       {test_accuracies_baseline[-1]:.2f}%")
    print(f"QI without Jet Loss:     {test_accuracies_no_jet[-1]:.2f}%")
    print(f"QI with Jet Loss:        {test_accuracies_jet[-1]:.2f}%")
    print("="*60)


def main():
    """Main training pipeline."""
    # Load data once
    train_loader, test_loader = load_coco_data()
    
    # Train all three models
    train_losses_baseline, test_accuracies_baseline = train_baseline_resnet(train_loader, test_loader)
    train_losses_no_jet, test_accuracies_no_jet = train_qi_model(train_loader, test_loader, use_jet_loss=False)
    train_losses_jet, test_accuracies_jet = train_qi_model(train_loader, test_loader, use_jet_loss=True)
    
    # Save and plot results
    save_and_plot_results(
        train_losses_baseline, test_accuracies_baseline,
        train_losses_no_jet, test_accuracies_no_jet,
        train_losses_jet, test_accuracies_jet
    )


if __name__ == "__main__":
    main()
