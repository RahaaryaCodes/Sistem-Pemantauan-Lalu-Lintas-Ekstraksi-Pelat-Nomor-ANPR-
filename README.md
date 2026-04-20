# 🚦 Smart Traffic Monitoring & ANPR System

An end-to-end Computer Vision pipeline designed for dynamic vehicle detection, bidirectional traffic counting, and Automatic Number Plate Recognition (ANPR) in real-world scenarios.

This repository serves not only as a functional system but also as a **demonstration of the computer vision evolution**—transitioning from classical image processing techniques to modern Deep Learning architectures.

---

## 🌟 Key Features

* **Real-Time Vehicle Detection:** Accurately identifies and classifies vehicles (cars, motorcycles, buses, trucks) using a finely tuned `YOLOv8` model.
* **Robust Object Tracking:** Integrates `ByteTrack` to assign and maintain unique vehicle IDs across frames, effectively handling object occlusion and fast-moving targets.
* **Bidirectional Counting (IN/OUT):** Implements dynamic virtual tripwires to analyze vehicle trajectories and count traffic flow in both directions based on geometric center-point analysis.
* **Specialized License Plate Detection:** Utilizes a custom-trained YOLO model specifically optimized for detecting small, dense objects (license plates) within the primary vehicle bounding boxes.
* **Optical Character Recognition (OCR):** Extracts alphanumeric characters from detected plates dynamically using `EasyOCR`, with custom regex filtering for Indonesian plate formats.
* **Nearest-Neighbor Association:** Algorithms to accurately map detected license plates to their respective vehicle IDs using Euclidean distance calculations.

---

## 🧠 The Evolution Pipeline (From Classical to Deep Learning)

To build a robust system, it is crucial to understand the foundational algorithms. This project was developed in stages to benchmark classical techniques against modern AI:

### Stage 1: Classical Computer Vision (The Baseline)
* **Approach:** Utilized `Haar Cascade` Classifiers and sliding window techniques.
* **Preprocessing:** Applied Grayscale conversion, Gaussian Blur, and CLAHE (Contrast Limited Adaptive Histogram Equalization).
* **Limitations Found:** Highly sensitive to lighting changes, camera angles, and prone to high false-positive rates. Overlapping boxes required manual Non-Maximum Suppression (NMS) tuning.

### Stage 2: Modern Deep Learning (The Solution)
* **Approach:** Migrated to `YOLOv8` (You Only Look Once) for automatic feature extraction and end-to-end learning.
* **Advantages:** Achieved significant improvements in accuracy, generalization across diverse lighting conditions, and real-time processing speeds.

---

## 🛠️ Tech Stack & Libraries

* **Core AI/ML:** Ultralytics YOLOv8, ByteTrack (Tracking)
* **Computer Vision:** OpenCV (`cv2`)
* **OCR Engine:** EasyOCR
* **Data Handling & Math:** NumPy, Math
* **Language:** Python

<img width="743" height="588" alt="Screenshot 2026-04-20 094744" src="https://github.com/user-attachments/assets/d04ea1e6-679c-46c6-9736-8313077fc484" />
<img width="1919" height="1079" alt="Screenshot 2026-04-20 094852" src="https://github.com/user-attachments/assets/fb5876d1-6273-4cf1-9eea-d0ad40f25c42" />
<img width="1900" height="1077" alt="Screenshot 2026-04-20 094858" src="https://github.com/user-attachments/assets/757cdc16-3417-4f5a-966c-dc24bd9afd26" />
<img width="1919" height="892" alt="image" src="https://github.com/user-attachments/assets/942e0f0e-e125-4493-8141-d52a2937a87d" />


---
