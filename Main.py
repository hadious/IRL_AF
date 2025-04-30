import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt
import numpy as np

# Simple CNN Backbone
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)

# Siamese network with two heads
class IRL_AF(nn.Module):
    def __init__(self, backbone, feature_dim=64):
        super().__init__()
        self.backbone = backbone
        self.invariance_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.equivariance_head = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Predicting rotation (0,90,180,270)
        )

    def forward(self, x):
        feat = self.backbone(x)
        z = self.invariance_head(feat)
        pred_rot = self.equivariance_head(feat)
        return z, pred_rot

# Data augmentations
def get_augmented_view(img, rotation):
    return transforms.functional.rotate(img, rotation)

# Prepare MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = IRL_AF(CNNBackbone()).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(5):  # just 5 epochs for demo
    model.train()
    total_loss, inv_loss_avg, eq_loss_avg = 0, 0, 0
    for imgs, _ in train_loader:
        optimizer.zero_grad()

        # Generate two views
        rotations = torch.randint(0, 4, (imgs.size(0),)) * 90
        imgs_view1 = torch.stack([get_augmented_view(img, rotations[i].item()) for i, img in enumerate(imgs)])
        imgs_view2 = torch.stack([get_augmented_view(img, 0) for img in imgs])

        imgs_view1, imgs_view2 = imgs_view1.to(device), imgs_view2.to(device)

        # Forward pass
        z1, pred_rot = model(imgs_view1)
        with torch.no_grad():
            z2, _ = model(imgs_view2)

        inv_loss = (1 - F.cosine_similarity(z1, z2.detach())).mean()

        eq_loss = F.cross_entropy(pred_rot, rotations.to(device)//90)

        # Total loss
        loss = inv_loss + eq_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        inv_loss_avg += inv_loss.item()
        eq_loss_avg += eq_loss.item()

    print(f"Epoch [{epoch+1}/5]: Total Loss={total_loss/len(train_loader):.4f}, Invariance={inv_loss_avg/len(train_loader):.4f}, Equivariance={eq_loss_avg/len(train_loader):.4f}")

# Extract embeddings and visualize using UMAP
model.eval()
embeddings, labels = [], []
with torch.no_grad():
    for imgs, lbls in train_loader:
        imgs = imgs.to(device)
        z, _ = model(imgs)
        embeddings.append(z.cpu().numpy())
        labels.append(lbls.numpy())

embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)

reducer = umap.UMAP()
umap_embedding = reducer.fit_transform(embeddings)

plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=labels, cmap='Spectral', s=1)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of MNIST embeddings')
plt.show()

# Linear classifier evaluation
linear_clf = nn.Linear(64, 10).to(device)
optimizer_clf = optim.Adam(linear_clf.parameters(), lr=1e-3)
criterion_clf = nn.CrossEntropyLoss()

for epoch in range(5):
    linear_clf.train()
    total_clf_loss = 0
    correct, total = 0, 0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        with torch.no_grad():
            z, _ = model(imgs)
        preds = linear_clf(z)
        loss = criterion_clf(preds, lbls)

        optimizer_clf.zero_grad()
        loss.backward()
        optimizer_clf.step()

        total_clf_loss += loss.item()
        correct += (preds.argmax(dim=1) == lbls).sum().item()
        total += lbls.size(0)

    accuracy = correct / total
    print(f"Linear classifier Epoch [{epoch+1}/5]: Loss={total_clf_loss/len(train_loader):.4f}, Accuracy={accuracy*100:.2f}%")
