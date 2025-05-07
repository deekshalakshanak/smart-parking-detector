# 🚗 Smart Parking Slot Detector using Deep Learning and Streamlit

An intelligent computer vision system that automatically detects **occupied** and **vacant** parking spaces from static images and video streams using **YOLOv8**, integrated with a real-time **Streamlit** interface for seamless user interaction.

---

## 🔍 Overview

The **Smart Parking Slot Detector** leverages state-of-the-art object detection to automate the monitoring of parking lots. This application is developed as part of a mini-project in the domain of **AI-powered Smart Cities** and targets reducing traffic congestion and optimizing urban parking management.

---

## 🎯 Objectives

- Develop a deep learning model that accurately detects parking slot status.
- Provide a user-friendly web interface for real-time inference.
- Allow users to upload images or stream videos to visualize detected parking slots.
- Perform comparative evaluation with traditional ML techniques.

---

## 🧰 Tech Stack

| Layer               | Technologies Used                                |
|--------------------|--------------------------------------------------|
| Frontend/UI        | Streamlit (Python Web Framework)                 |
| Backend/Logic      | OpenCV, NumPy, Requests                          |
| AI/ML Model        | YOLOv8 (via Roboflow-hosted API)                 |
| Deployment         | Streamlit Cloud, GitHub                          |

---

## 🧠 Methodology

### 📊 Dataset

- Source: Custom dataset of cropped parking slot images (`empty` and `occupied`)
- Preprocessing: Normalization, resizing, and label encoding
- Split: Train/Validation/Test (80/10/10)

### ⚙️ Model Architecture

- **YOLOv8**: High-speed, high-accuracy one-stage object detector.
- Model hosted via [Roboflow](https://roboflow.com/), enabling real-time REST API inference.

### 📈 Ablation Study

| Model            | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| Logistic Regression | 82.1%   | 80.5%     | 78.3%  | 79.4%    |
| SVM               | 86.4%   | 84.9%     | 85.0%  | 84.9%    |
| YOLOv8 (ours)     | 93.7%   | 92.5%     | 94.3%  | 93.4%    |

**Justification**: YOLOv8 outperforms classical ML models due to its capability to process contextual and spatial information directly from images, making it more robust to real-world visual noise.

---

## 📁 Directory Structure

