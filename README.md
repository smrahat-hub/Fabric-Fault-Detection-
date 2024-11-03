# YOLOv8 Fabric Defect Detection

This project uses YOLOv8 to train a custom model for detecting defects on fabric images and videos. The model identifies and annotates specific defects on fabrics, saving the processed images and videos with annotations to Google Drive.

## Table of Contents

- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Image Detection](#image-detection)
- [Video Detection](#video-detection)
- [Notes](#notes)

---

## Installation

1. **Mount Google Drive**  
   Ensure your Google Drive is mounted to save and load data.

   ```python
   from google.colab import drive
   drive.mount('/content/gdrive')
2. **Install Ultralytics**  
    Install the Ultralytics library to use YOLOv8.

   ```python
   !pip install ultralytics

# Training the Model
Load the YOLOv8 model and train it on a custom dataset for fabric defect detection.

```python
from ultralytics import YOLO
import os

# Define the root directory for your dataset
ROOT_DIR = '/content/gdrive/My Drive/Model_train/data'

# Load and train the model
model = YOLO('yolov8n.pt')
results = model.train(data=os.path.join(ROOT_DIR, "config.yaml"), epochs=100)

# Save the trained model
!scp -r /content/runs '/content/gdrive/My Drive/Model_train/data'
```

# Image Detection
This section demonstrates how to load a trained model to perform predictions on images.

1. **Load the Model**
Specify the path to your trained model.

```python
model = YOLO('/content/gdrive/My Drive/Model_train/data/runs/detect/train2/weights/best.pt')
```
2. **Run Detection on Uploaded Images**
Use the detect_detection function to predict defects and save annotated images to Google Drive.

```python
from google.colab.patches import cv2_imshow
from google.colab import files
import cv2

uploaded = files.upload()

for filename in uploaded.keys():
    detect_detection(filename)

```

# Video Detection
To detect defects in videos, upload a video file and process each frame. The annotated video is saved back to Google Drive
1. **Load the Model**
Specify the path to your trained model.

```python
model = YOLO('/content/gdrive/My Drive/Model_train/data/runs/detect/train2/weights/last.pt')
```
2. **Process Uploaded Videos**
Use the process_video function to annotate defects frame-by-frame in the uploaded video.

```python
uploaded = files.upload()

for video in uploaded.keys():
    process_video(video)


```
