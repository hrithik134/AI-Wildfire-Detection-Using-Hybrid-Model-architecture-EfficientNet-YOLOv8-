import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, epochs=30):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1} Validation Accuracy: {acc:.4f}")

    return model

# Main Function Block
if __name__ == '__main__':
    # Dataset and DataLoaders
    train_dataset = FireDataset('D://AI wildfire//dataset-classification//train//labels.csv', 'D://AI wildfire//dataset-classification//train//images', train_transform)
    val_dataset   = FireDataset('D://AI wildfire//dataset-classification//val//labels.csv',   'D://AI wildfire//dataset-classification//val//images', val_test_transform)
    test_dataset  = FireDataset('D://AI wildfire//dataset-classification//test//labels.csv.', 'D://AI wildfire//dataset-classification//test//images', val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Model Setup
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    class_counts = [5000, 300, 1700]  # Fire, Smoke, Non-Fire
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    print(f"Class Weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.amp.GradScaler()  # Updated as per warning

    # Train
    start = time.time()
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, epochs=30)
    end = time.time()
    print(f"Total training time: {(end-start)/60:.2f} minutes")

    # Save Model
    torch.save(trained_model.state_dict(), "efficientnet_fire_smoke.pt")

    # Test
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())

    print("Test Accuracy:", accuracy_score(test_labels, test_preds))
    print("Classification Report:\n", classification_report(
    test_labels, test_preds,
    labels=[0, 1, 2],  # 0=Fire, 1=Smoke, 2=Non-Fire
    target_names=['Fire', 'Smoke', 'Non-Fire'],
    zero_division=0   # To suppress division errors if a class is missing
))
