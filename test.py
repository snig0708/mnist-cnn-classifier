from pathlib import Path
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchmetrics import Precision, Recall, F1Score

metric_precision = Precision(task = "multiclass", num_classes = 10, average = "macro" )
metric_recall = Recall(task = "multiclass", num_classes = 10, average = "macro" )
metric_f1 = F1Score(task = "multiclass", num_classes = 10, average = "macro" )


class Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net(num_classes=10).to(device)
net.load_state_dict(torch.load("model_state.pt", map_location=device))
net.eval()


with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = net(images)
        _,preds = torch.max(outputs, 1)
        metric_precision(preds, labels)
        metric_recall(preds, labels)
        metric_f1(preds, labels)

precision = metric_precision.compute()
recall = metric_recall.compute()
f1 = metric_f1.compute()

print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test f1: {f1:.4f}")

# Single custom image: mnist_classification/image.jpg
image_path = Path(__file__).resolve().parent / "image.jpg"
img_transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)

if image_path.is_file():
    img = Image.open(image_path).convert("L")
    img_tensor = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = net(img_tensor)
        digit = logits.argmax(dim=1).item()
    print(f"\nCustom image ({image_path.name}): predicted digit {digit}")
else:
    print(
        f"\nNo file at {image_path} — place image.jpg there to run custom inference."
    )
