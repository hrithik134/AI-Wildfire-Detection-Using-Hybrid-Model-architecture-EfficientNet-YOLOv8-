import os
import shutil
import random
import pandas as pd

random.seed(42)

input_folder = "D://AI wildfire//dataset-conv"
output_folder = "D://AI wildfire//dataset-classification"

splits = ['train', 'val', 'test']
split_ratio = [0.7, 0.2, 0.1]  # 70% train, 20% val, 10% test
class_map = {'fire': 0, 'smoke': 1, 'non_fire': 2}

# Create folders
for split in splits:
    os.makedirs(os.path.join(output_folder, split, 'images'), exist_ok=True)

# Collect all data and split
for cls_name, cls_idx in class_map.items():
    cls_path = os.path.join(input_folder, cls_name)
    images = [img for img in os.listdir(cls_path) if img.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    train_split = int(split_ratio[0] * len(images))
    val_split = int(split_ratio[1] * len(images))

    split_data = {
        'train': images[:train_split],
        'val': images[train_split:train_split+val_split],
        'test': images[train_split+val_split:]
    }

    for split in splits:
        rows = []
        for img_name in split_data[split]:
            src = os.path.join(cls_path, img_name)
            dst = os.path.join(output_folder, split, 'images', img_name)
            shutil.copy(src, dst)
            rows.append({'filename': img_name, 'label': cls_idx})

        # Write CSV
        csv_path = os.path.join(output_folder, split, 'labels.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        else:
            df = pd.DataFrame(rows)

        df.to_csv(csv_path, index=False)

print("✅ Dataset split and CSV creation complete!")
