import warnings
import os
import logging
from ultralytics import YOLO

# Suppress Python warnings
warnings.filterwarnings("ignore")

# Suppress Ultralytics and other loggers
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # For TensorFlow if it's present
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

def train_yolo():
    model = YOLO("yolov8m.pt")  # Load a pre-trained YOLOv8 model

    model.train(
        data=r"D:\AI wildfire\data.yaml",  # Path to your dataset YAML
        epochs=50,
        patience=10,         # Early stopping if no improvement
        imgsz=640,           # Image size
        batch=16,            # Batch size
        device=0,            # GPU ID
        amp=True,            # Enable Automatic Mixed Precision
        workers=4,           # Dataloader workers (adjust as needed)
        optimizer="AdamW",   # AdamW for better generalization
        lr0=0.001,           # Learning rate
        verbose=False        # Disable verbose logging during training
    )

if __name__ == "__main__":
    train_yolo()
