import pandas as pd
import matplotlib.pyplot as plt

# Load labels CSV (update path if needed)
df = pd.read_csv("dataset-classification/train/labels.csv")

# Count class distribution
class_counts = df['label'].value_counts().sort_index()

# Match labels with indexes dynamically
label_names = {0: 'Fire', 1: 'Smoke', 2: 'Non-Fire'}
class_labels = [label_names.get(i, f"Class {i}") for i in class_counts.index]

# Plot
plt.figure(figsize=(6, 4))
plt.bar(class_labels, class_counts.values, color=['red', 'gray', 'green'][:len(class_counts)])
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Figure 2. Class distribution in the dataset used for training and evaluation')
plt.tight_layout()
plt.savefig('figure2_class_distribution.png')
plt.show()
