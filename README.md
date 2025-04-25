📱 Smartphone Detection Using YOLOv8
🚨 Problem Statement
The objective of this project is to develop a robust real-time smartphone detection system using the YOLOv8 object detection framework. Detecting smartphone usage, especially in restricted environments like classrooms, driving scenarios, or workplaces, can enhance safety and productivity.

📘 Introduction
Smartphone usage in sensitive or secure environments can be a cause for concern. With the growing accessibility of deep learning, real-time object detection offers a powerful way to automate monitoring. This project leverages YOLOv8 to detect whether a person is holding or using a mobile phone from various input sources like images, videos, and webcam feeds.

⚙️ Working
This repository includes three primary detection scripts:

detect_on_image.py: Detect smartphones in a static image.
detect_phone_on_video.py: Run detection on a pre-recorded video.
live_phone_detection.py: Detect phones in real-time using your webcam.
The trained model (best.pt) is stored in the Models/ directory and is used by all scripts for inference. The model was trained on a custom dataset and optimized for real-world smartphone detection scenarios.

📊 Results
The model has been evaluated using various metrics. Below are visualizations of its performance:

🔸 Precision-Recall and F1 Score Curves
PR Curve F1 Curve

🔸 Confusion Matrix
Confusion Matrix

🔸 Class Distribution & Additional Analysis
Labels Correlogram

🧪 How to Run This on Your Computer
🔧 Requirements
Python 3.8+
PyTorch
OpenCV
Ultralytics YOLOv8 (install via pip)
pip install ultralytics opencv-python
📂 Clone the Repo
git clone https://github.com/Aarya108/Smartphone-Detection-YOLO.git
cd Smartphone-Detection-YOLO
📁 Project Structure
Smartphone-Detection-YOLO/
│
├── Models/
│   └── best.pt                # YOLOv8 trained weights
├── notebooks/                 # (for training/EDA - optional)
├── results/                   # Evaluation and performance visuals
│
├── detect_on_image.py
├── detect_phone_on_video.py
├── live_phone_detection.py
├── requirements.txt           # Optional: dependencies list
└── README.md
▶️ Run the Scripts
1. Image Detection
python detect_on_image.py --source path/to/image.jpg --weights Models/best.pt
2. Video Detection
python detect_phone_on_video.py --source path/to/video.mp4 --weights Models/best.pt
3. Real-time Webcam Detection
python live_phone_detection.py --weights Models/best.pt
🧠 Model Details
Architecture: YOLOv8-medium
Dataset: Custom-labeled images with positive (phone) and negative samples
Training platform: Google Colab
