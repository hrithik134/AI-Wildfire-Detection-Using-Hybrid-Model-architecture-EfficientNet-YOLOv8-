from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

# Load an image
image_path = "D://AI wildfire//firesmoke1.jpg"  # replace with actual file
img = Image.open(image_path).convert("RGB")

# Resize and normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0,1]
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
img_tensor = transform(img)

# Unnormalize for visualization
unnorm = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)
img_unnorm = unnorm(img_tensor).permute(1, 2, 0).clamp(0, 1).numpy()

# Plot
plt.imshow(img_unnorm)
plt.title("Figure 3. Example of a resized and normalized image for EfficientNet-B0 input")
plt.axis("off")
plt.tight_layout()
plt.savefig("figure3_resized_normalized_image.png")
plt.show()
