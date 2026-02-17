
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Dataset parameters
NUM_CLASSES = 10  # MNIST has 10 digit classes
BATCH_SIZE = 512
EPOCHS = 30
LR = 0.001

K_PROBES = 28
EPSILON = 0.1
LAMBDA_JET = 0.1      # Weight for alignment in Jet Loss
ETA_JET = 0.5         # Weight of Jet Loss in total loss [cite: 82]

# Device setup for CUDA
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class MNISTClassifier(nn.Module):
    def __init__(self, num_classes, k_probes, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.k_probes = k_probes
        self.num_classes = num_classes

        v = torch.randn(k_probes, 28*28)  # [k_probes, 784]
        v = v / torch.norm(v, dim=1, keepdim=True)
        self.register_buffer('probes', v)

        # ResNet18 backbone (modified for grayscale MNIST)
        resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        
        # Modify first conv layer for single channel input
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
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=1, bias=True)
        )

        self.model_lower = nn.Sequential(
            nn.Conv2d(128 + 3, 128, kernel_size=1, padding=0, bias=True),  # 128 from ResNet + O0, O1, O2
            resnet_model.layer3,
            resnet_model.layer4,
            resnet_model.avgpool,
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes, bias=True)
        )

    def compute_score(self, x):
        if not x.requires_grad:
            x.requires_grad_(True)

        # Compute energy as mean of representation
        h = self.model_upper(x)
        energy = self.scalar_projection(h)

        grads = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]

        # grads shape: [batch, 1, 28, 28], probes shape: [k_probes, 28, 28]
        # Flatten spatial dimensions for dot product
        grads_flat = grads.view(grads.size(0), -1)  # [batch, 784]
        probes_flat = self.probes.view(self.k_probes, -1)  # [k_probes, 784]
        
        scores = -torch.matmul(grads_flat, probes_flat.T)  # [batch, k_probes]
        return scores

    def compute_local_fisher(self, x):
        scores = self.compute_score(x)
        fisher_info = (scores ** 2).mean(dim=1, keepdim=True)
        return fisher_info

    def forward(self, x):
        with torch.set_grad_enabled(True):
            if not x.requires_grad:
                x.requires_grad_(True)

            # Compute derivatives with respect to input
            O0 = self.compute_local_fisher(x)

            # For image tensors, use small Gaussian perturbations instead of probe directions
            mean_v = self.probes.mean(dim=0, keepdim=True).view(1, 1, 28, 28)  # [1, 1, 28, 28]
            x_pos = x + mean_v
            x_neg = x - mean_v

            I_pos = self.compute_local_fisher(x_pos)
            I_neg = self.compute_local_fisher(x_neg)

            O1 = (I_pos - I_neg) / (2 * self.epsilon)
            O2 = (I_pos - 2 * O0 + I_neg) / (self.epsilon ** 2)

        # Forward through ResNet
        h = self.model_upper(x)  # h shape: [batch, 128, 4, 4]
        
        # Reshape O0, O1, O2 to [batch, 1, 1, 1] and broadcast to [batch, 1, 4, 4]
        O0_broadcast = O0.view(-1, 1, 1, 1).expand(-1, 1, h.size(2), h.size(3))  # [batch, 1, 4, 4]
        O1_broadcast = O1.view(-1, 1, 1, 1).expand(-1, 1, h.size(2), h.size(3))  # [batch, 1, 4, 4]
        O2_broadcast = O2.view(-1, 1, 1, 1).expand(-1, 1, h.size(2), h.size(3))  # [batch, 1, 4, 4]
        
        # Concatenate all features
        features = torch.cat([h, O0_broadcast, O1_broadcast, O2_broadcast], dim=1)  # [batch, 131, 4, 4]

        y_hat = self.model_lower(features)
        
        return y_hat, O0 # Return O0 for viz/loss if needed



# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

print("Loading MNIST dataset...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


# Create model
model = MNISTClassifier(NUM_CLASSES, K_PROBES, EPSILON)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

train_losses = []
test_accuracies = []

print("Starting MNIST Training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output, energy_score = model(data)
        loss = criterion(output, target)
        mean_v = model.probes.mean(dim=0, keepdim=True).view(1, 1, 28, 28)

        # (Uncomment below to enable Jet Loss - adds compute time)
        x_pos = data + EPSILON * mean_v
        x_neg = data - EPSILON * mean_v
        pred_pos, _ = model(x_pos)
        pred_neg, _ = model(x_neg)
        D_hat = (pred_pos - pred_neg) / (2 * EPSILON) # [cite: 75]
        score_proj = model.compute_score(data).mean(dim=1, keepdim=True)  # [batch, 1]
        loss_jet = ((D_hat + LAMBDA_JET * score_proj)**2).mean() #
        loss += ETA_JET * loss_jet


        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Evaluate on test set
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

# Plot training results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('MNIST Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(test_accuracies, label='Test Accuracy', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('MNIST Test Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mnist_training_Jet_Loss.png', dpi=150, bbox_inches='tight')
print("Training plots saved as 'mnist_training_Jet_Loss.png'")
plt.show()

