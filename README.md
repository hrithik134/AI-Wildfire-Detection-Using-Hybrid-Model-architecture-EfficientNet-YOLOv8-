## AI Wildfire Detection Using EfficientNet and YOLOv8

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

This project implements a **real-time wildfire detection system** using a combination of deep learning models:
- **EfficientNet-B0** for **image classification** (detecting fire, smoke, or non-fire).
- **YOLOv8** for **bounding box object detection** of fire and smoke regions in real-time.
- A simple **alarm system** is included to alert users when fire or smoke is detected. The alarm logic is implemented in the code and can be customized as needed.

The goal of this project is to leverage AI for **early wildfire detection** using surveillance footage, drone imagery, or camera feeds, aiming to prevent and mitigate fire-related damages.

---

## Model Pipeline

### 1. Image Classification (EfficientNet-B0)

- The first stage classifies input frames into three classes: `fire`, `smoke`, and `non-fire`.
- It uses **EfficientNet-B0**, a lightweight and efficient convolutional neural network, for better accuracy and performance.

### 2. Object Detection (YOLOv8)

- Once fire or smoke is detected, the second stage triggers **YOLOv8** (You Only Look Once) to draw bounding boxes around the detected fire or smoke areas in real-time.
- YOLOv8 is a state-of-the-art model for object detection, providing real-time localization of fire regions.

---

## Results & Example Outputs

### Example 1: Fire Detection

- **Input**: A frame of a wildfire.
- **Output**: Bounding boxes drawn around fire regions.

### Example 2: Smoke Detection

- **Input**: A frame containing smoke.
- **Output**: Bounding boxes drawn around smoke regions.

### Example Output Images

![YOLOv8 Output](detcted%20imgs/yolo_detected.jpg)  
![YOLOv8 Output](detcted%20imgs/yolo_detected1.png)  
![YOLOv8 Output](detcted%20imgs/yolo_detected2.png)  
![YOLOv8 Output](detcted%20imgs/yolo_detected3.png)

---

## Technologies Used

- **Programming Languages**: Python
- **Frameworks**: PyTorch, OpenCV, YOLOv8
- **Libraries**: EfficientNet, NumPy, Matplotlib, Pandas
- **Tools**: CUDA (GPU acceleration), Google Colab (for training)
- **Miscellaneous**: Tkinter (GUI for real-time inference), Sound-based alarm

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-wildfire-detection.git
cd ai-wildfire-detection
```

### 2. Install Dependencies
Make sure you have Python 3.6 or higher installed, and then use the following to install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Download Pretrained Weights
Download the EfficientNet-B0 and YOLOv8 weights (trained models) and place them in the models/ directory.

##### Note: You can use publicly available fire and smoke datasets (such as FireNet, FiSmo, or Kaggle wildfire datasets) to train the YOLOv8 model. The same dataset (after converting it to image-label format) can also be used to train the EfficientNet-B0 classification model by creating labeled folders such as fire/, smoke/, and non-fire/.

### 4. Running the Project
Static Image Detection :

To run the Static Image detection system, execute:

```bash
python inference-eff.py
```

Real-Time Detection :

To run the real-time detection system, execute:

```bash
python realtime_inference.py
```

### Training Models
If you'd like to train the models from scratch or fine-tune them:

Train EfficientNet-B0:

```bash
python train_effnet.py
```

Train YOLOv8:

```bash
python train_yolov8.py
```

### Contributors
Hrithik (Project Lead): [GitHub Profile](https://github.com/hrithik134)

Vignesh : https://github.com/rockyvicky123

### 📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Future Improvements
1. Multi-camera Integration: Expand the system to handle multiple video feeds for broader coverage.

2. Smoke vs Fire Differentiation: Improve classification by differentiating between smoke and fire more accurately.

3. Web Interface: Build a web interface for real-time visualization and data analytics.
 
4. Heat Image Mapping
