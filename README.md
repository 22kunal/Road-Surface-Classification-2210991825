# Road Surface Classification
### Modern Approaches to Road Surface Classification: Assessing CNN vs. KNN Models

**Student:** Kunal Kumar  
**Roll No:** 2210991825  
**Type:** Research Paper  
**Supervisor:** Dr. Lalit K Sharma  
**Department:** Computer Science and Engineering, Chitkara University  
**Current Status:** Evaluation 3 Submission Complete  

---

## 🚀 Live Demo

**Try the working web app here (permanent link):**

[![Hugging Face](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-blue)](https://huggingface.co/spaces/Kunal2226/road-surface-classifier)

🔗 https://huggingface.co/spaces/Kunal2226/road-surface-classifier

> Point your webcam at a road surface OR upload a road image.
> The CNN+KNN model will classify it as Dry / Wet / Muddy in real time.

---

## Run the Code in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/22kunal/Road-Surface-Classification-2210991825/blob/main/Source%20Code/road_classification_2210991825.ipynb)

---

## Project Overview

A hybrid CNN-KNN model for classifying road surfaces into three categories —
**Dry, Wet, and Muddy** — using deep learning and instance-based classification.

The CNN extracts visual features from road surface images through 4 convolutional
layers and 4 max-pooling layers. These features are then passed to a KNN classifier
for the final prediction. The model achieved **98.31% accuracy** on 10,680 images,
outperforming LSTM, Random Forest, ResNet50, and SVM baselines.

---

## Repository Structure
Road-Surface-Classification-2210991825/
│
├── Report and PPT/
│   ├── project_report.docx
│   └── project_presentation.pptx
│
├── Source Code/
│   └── road_classification_2210991825.ipynb
│
└── README.md

---

## Problem Statement

Road surface conditions (Dry, Wet, Muddy) directly affect vehicle safety and
performance. Autonomous vehicles and ADAS systems need real-time road surface
awareness to adjust braking distance, traction control, and steering sensitivity.
This project proposes a hybrid CNN-KNN approach that is accurate, automated,
and computationally efficient.

---

## Methodology

| Phase | Description |
|-------|-------------|
| Phase 1 | Data preparation — images, augmentation, 80/20 split |
| Phase 2 | CNN feature extraction — 4 Conv2D + 4 MaxPool + Dropout + Dense(128) |
| Phase 3 | KNN classification — FC layer features → KNeighborsClassifier |
| Phase 4 | Evaluation — Accuracy, Precision, Recall, F1, Confusion Matrix |

---

## Results

| Model | Accuracy |
|-------|----------|
| **CNN-KNN Hybrid (Proposed)** | **98.31%** |
| Random Forest | 93.44% |
| ResNet50 | 91.06% |
| LSTM | 90.56% |
| SVM | 89.58% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Dry Road | 98.36% | 98.36% | 98.36% |
| Wet Road | 98.31% | 98.31% | 98.31% |
| Muddy Road | 98.27% | 98.27% | 98.27% |

---

## How to Run Locally

1. Click **Open in Colab** button above
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Click **Runtime → Run all**
4. Wait ~20 minutes for training to complete

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.10 | Programming language |
| TensorFlow 2.x / Keras | CNN model building and training |
| Scikit-learn | KNN classifier and evaluation metrics |
| Matplotlib / Seaborn | Graphs and confusion matrix |
| Google Colab (T4 GPU) | Cloud training environment |
| Gradio | Web application framework |
| Hugging Face Spaces | Permanent deployment platform |

---

## Live Application

The trained model is deployed as a web application on Hugging Face Spaces.
Features:
- 🎥 Real-time webcam classification
- 📁 Image upload support  
- 📊 Confidence scores for all 3 classes
- ⚠️ Safety warnings based on road condition
- 🌐 Accessible from any browser, any device

🔗 **https://huggingface.co/spaces/Kunal2226/road-surface-classifier**
