import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from torch.utils.data import DataLoader

# Assuming 'device' is your device (cuda or cpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming 'model' is your trained model and loaded onto the device
model = YourModelClass()  # Initialize your model here
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Assuming 'val_loader' is your validation DataLoader (make sure it's initialized correctly)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Define your DataLoader here

# Initialize lists to store true and predicted labels
all_labels = []
all_preds = []

# Assuming you're using a validation DataLoader named 'val_loader'
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Get model predictions
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)

        # Append true labels and predictions to the lists
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Define the class labels (update as per your project)
classes = ['Fire', 'Smoke', 'Non-Fire']

# 1. Normalized Confusion Matrix
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2], normalize='true')
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap='Oranges', xticklabels=classes, yticklabels=classes)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 2. Confidence Distribution Histogram
softmax = torch.nn.Softmax(dim=1)
confidences = []

with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = softmax(outputs)
        max_confidences, _ = torch.max(probs, dim=1)
        confidences.extend(max_confidences.cpu().numpy())

plt.figure(figsize=(7, 4))
plt.hist(confidences, bins=30, color='skyblue', edgecolor='black')
plt.title("Model Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3. Per-Class Accuracy Bar Chart
cm_raw = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
per_class_acc = cm_raw.diagonal() / cm_raw.sum(axis=1)

plt.figure(figsize=(6, 4))
plt.bar(classes, per_class_acc, color=['#4caf50', '#ff9800', '#2196f3'])
plt.title("Per-Class Accuracy")
plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()