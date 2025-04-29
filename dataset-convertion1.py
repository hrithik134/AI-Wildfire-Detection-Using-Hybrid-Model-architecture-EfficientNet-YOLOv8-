import os
import shutil

# Paths
base_path = "D://AI wildfire//dataset//train"
images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")

output_base = "D://AI wildfire//dataset-conv"
classes = ["fire", "smoke", "non_fire"]

# Create output folders
for cls in classes:
    os.makedirs(os.path.join(output_base, cls), exist_ok=True)

# Process each image
for filename in os.listdir(images_path):
    if not filename.endswith(('.jpg', '.png', '.jpeg')):
        continue

    name = os.path.splitext(filename)[0]
    image_file = os.path.join(images_path, filename)
    label_file = os.path.join(labels_path, name + ".txt")

    target_class = "non_fire"  # default

    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            if lines:
                first_class = int(lines[0].split()[0])
                if first_class == 0:
                    target_class = "fire"
                elif first_class == 1:
                    target_class = "smoke"

    shutil.copy(image_file, os.path.join(output_base, target_class, filename))

print("✅ Dataset conversion complete!")