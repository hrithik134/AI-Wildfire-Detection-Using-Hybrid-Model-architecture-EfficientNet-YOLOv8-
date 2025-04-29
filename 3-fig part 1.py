import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Class
class FireDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None):
        self.labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.labels.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load validation data
val_dataset = FireDataset('D:/AI wildfire/dataset-classification/val/labels.csv',
                          'D:/AI wildfire/dataset-classification/val/images',
                          transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load model
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
model.load_state_dict(torch.load("D:/AI wildfire/weights/efficientnet_fire_smoke.pt", map_location=device))
model.to(device)
model.eval()

# Run inference
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
print("Accuracy:", acc)
report = classification_report(
    all_labels, all_preds,
    labels=[0, 1, 2],  # Explicitly specify all 3 classes
    target_names=['Fire', 'Smoke', 'Non-Fire'],
    digits=3, zero_division=0
)
print(report)

# Confusion Matrix Heatmap
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fire', 'Smoke', 'Non-Fire'], yticklabels=['Fire', 'Smoke', 'Non-Fire'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Bar chart for precision, recall, f1-score
report_dict = classification_report(
    all_labels, all_preds,
    labels=[0, 1, 2],
    target_names=['Fire', 'Smoke', 'Non-Fire'],
    output_dict=True,
    zero_division=0
)

metrics = ['precision', 'recall', 'f1-score']
classes = ['Fire', 'Smoke', 'Non-Fire']

values = {metric: [report_dict[c][metric] for c in classes] for metric in metrics}

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(8, 5))
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, values[metric], width=width, label=metric)

plt.xticks(x + width, classes)
plt.ylim(0, 1.1)
plt.ylabel("Score")
plt.title("Precision, Recall, F1-Score")
plt.legend()
plt.tight_layout()
plt.show()