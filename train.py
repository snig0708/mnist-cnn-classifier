# Importing dependencies
import torch
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose(
    [
        transforms.RandomRotation(degrees=10),  # ±10° — light tilt typical of handwriting
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.MNIST(
    root="data",
    download=True,
    train=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(
    root="data",
    download=True,
    train=False,
    transform=test_transform,
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Define the image classifier model
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2), 
            nn.Conv2d(32, 64, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# Create an instance of the image classifier model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(num_classes=10).to(device)

# Define the optimizer and loss function
optimizer = Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(10):  # Train for 10 epochs
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = net(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()

    epoch_loss = running_loss/len(train_loader)
    print(f"Epoch:{epoch} loss is {epoch_loss:.4f}")

# Per-class recall on test set (recall_c = TP_c / support_c)
num_classes = 10
confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
net.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = net(images).argmax(dim=1).cpu()
        labels = labels.cpu()
        idx = labels * num_classes + preds
        confusion += torch.bincount(idx, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes
        )

row_sums = confusion.sum(dim=1).float()
diag = confusion.diag().float()
recall_per_class = torch.where(
    row_sums > 0,
    diag / row_sums,
    torch.full_like(diag, float("nan")),
)
for c in range(num_classes):
    r = recall_per_class[c].item()
    print(f"Class {c} recall: {r:.4f}" if r == r else f"Class {c} recall: n/a")

# Save the trained model
torch.save(net.state_dict(), 'model_state.pt')