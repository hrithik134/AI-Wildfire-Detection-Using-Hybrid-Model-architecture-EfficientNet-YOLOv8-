import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Assuming 'all_labels' and 'all_preds' are the true and predicted labels, 
# and 'val_loader' is your validation data loader
# Also, assuming 'device' is your device (cuda or cpu), and 'model' is your trained model

# Define the class labels
classes = ['Fire', 'Smoke', 'Non-Fire']  # Modify if needed

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
