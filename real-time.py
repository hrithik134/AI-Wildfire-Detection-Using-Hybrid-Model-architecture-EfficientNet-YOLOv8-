import cv2
from ultralytics import YOLO
import torch
import warnings
import requests  # For sending alert to the fire station (another laptop)

warnings.filterwarnings("ignore")

# Load trained YOLOv8 model
yolo_model = YOLO("best.pt")

# Force inference on CPU
torch_device = torch.device("cpu")
yolo_model.to(torch_device)

# Open webcam
cap = cv2.VideoCapture(0)

# Set resolution (optional for CPU speed-up)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

alert_sent = False  # To prevent repeated alerts

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 Inference
    results = yolo_model.predict(source=frame, conf=0.3, device="cpu", verbose=False)

    fire_detected = False  # Reset detection flag

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            # Set color for bounding box
            color = (0, 0, 255) if label.lower() == "fire" else (0, 255, 255)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Trigger Alarm if Fire Detected
            if label.lower() == "fire" and not alert_sent:
                fire_detected = True

    # Send alert only once fire detected
    if fire_detected and not alert_sent:
        try:
            response = requests.post("http://192.168.1.10:5000/alert", json={"status": "fire"})
            print("📡 Alert sent to fire station:", response.json())
            alert_sent = True  # Prevent spamming
        except Exception as e:
            print("Failed to send alert:", e)

    # Show live frame
    cv2.imshow("Real-Time Fire & Smoke Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close windows
cap.release()
cv2.destroyAllWindows()