from io import BytesIO
import warnings
import requests
import torch
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
import argparse
from torchvision import transforms, models
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# Helper function to load image from a URL
def img_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fire/Smoke Detection using EfficientNet + YOLOv8")
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load EfficientNet-B0 for classification
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 3)  # 3 classes: 0-Fire, 1-Smoke, 2-Non-Fire
    model.load_state_dict(torch.load("D://AI wildfire//weights//efficientnet_fire_smoke.pt", map_location=device))
    model = model.to(device)
    model.eval()

    # Load image
    if args.image:
        image_path = args.image
        image = Image.open(image_path).convert("RGB")
    else:
        image = img_from_url("https://s.abcnews.com/images/US/northern-california-fire-09-gty-jc-181109_hpMain_16x9_992.jpg")

    # Preprocess for EfficientNet
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image_tensor = transform_pipeline(image).unsqueeze(0).to(device)

    # Classification using EfficientNet
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1).item()
    labels = {0: "Fire", 1: "Smoke", 2: "Non-Fire"}

    print(f"\nEfficientNet Prediction: {labels[prediction]}")

    # If Fire or Smoke detected → Run YOLO and Trigger Alarm
    if prediction in [0, 1, 2]:
        print("YOLO activated to localize fire/smoke regions...")

        # 🚨 Send Alert to Other Laptop (Replace with actual IP of receiver)
        try:
            response = requests.post("http://172.20.10.3:5000/alert", json={"status": "fire"})
            print("Alert sent to fire station:", response.json())
        except Exception as e:
            print("Could not send alert to fire station:", e)

        yolo_model = YOLO("D://AI wildfire//best.pt")  # Use your trained YOLOv8 model

        # Convert image to OpenCV format if needed
        if args.image:
            source_input = args.image  # YOLO can take file path
        else:
            source_input = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL to OpenCV

        results = yolo_model.predict(source=source_input, save=False, conf=0.3)

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                print("No objects detected by YOLO.")
            else:
                print(f"YOLO detected {len(r.boxes)} object(s)")
                img_with_boxes = r.plot()
                cv2.imshow("YOLOv8 Detection", img_with_boxes)
                cv2.imwrite("D:\AI wildfire\detcted imgs\yolo_detected.jpg", img_with_boxes)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    else:
        # Show image with classification result
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        cv2.putText(image_cv, "No Fire or Smoke Detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("EfficientNet Detection", image_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()