# Road Surface Classification
### Modern Approaches to Road Surface Classification: Assessing CNN vs. KNN Models

**Student:** Kunal Kumar  
**Roll No:** 2210991825  
**Type:** Research Paper  
**Supervisor:** Dr. Lalit K Sharma  
**Department:** Computer Science and Engineering, Chitkara University  

---

## Project Overview

A hybrid CNN-KNN model for classifying road surfaces into three categories —
**Dry, Wet, and Muddy** — using deep learning and instance-based classification.

The CNN extracts visual features from road surface images through 4 convolutional
layers and 4 max-pooling layers. These features are then passed to a KNN classifier
for the final prediction. The model achieved **98.31% accuracy** on 10,680 images,
outperforming LSTM, Random Forest, ResNet50, and SVM baselines.

---

## Run the Code Directly in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/22kunal/Road-Surface-Classification-2210991825/blob/main/Source%20Code/road_classification_2210991825.ipynb)

> Click the button above to open the notebook directly in your browser.  
> No installation required.  
> Go to **Runtime → Change runtime type → T4 GPU → Run all**

---

## Repository Structure
Road-Surface-Classification-2210991825/
│
├── Report and PPT/
│   ├── project_report.docx          ← Full project report
│   └── project_presentation.pptx   ← Project presentation slides
│
├── Source Code/
│   └── road_classification_2210991825.ipynb  ← Colab notebook
│
└── README.md                        ← This file

---

## Problem Statement

Road surface conditions (Dry, Wet, Muddy) directly affect vehicle safety and
performance. Autonomous vehicles and ADAS systems need real-time road surface
awareness to adjust braking distance, traction control, and steering sensitivity.
Traditional methods are not scalable or robust enough for real-world deployment.
This project proposes a hybrid CNN-KNN approach that is accurate, automated, and
computationally efficient.

---

## Methodology

| Phase | Description |
|-------|-------------|
| Phase 1 | Data preparation — 10,680 images, 180x180px, augmentation, 80/20 split |
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

## How to Run

1. Click the **Open in Colab** button above
2. Go to **Runtime → Change runtime type → T4 GPU → Save**
3. Click **Runtime → Run all**
4. Wait ~20 minutes for dataset download and training
5. Results, graphs, confusion matrix saved to `results/` folder automatically
6. Last cell downloads a zip of all results to your computer

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.10 | Programming language |
| TensorFlow 2.x / Keras | CNN model building and training |
| Scikit-learn | KNN classifier and evaluation metrics |
| Matplotlib / Seaborn | Graphs and confusion matrix |
| Google Colab (T4 GPU) | Cloud training environment |
| Kaggle | Road surface image dataset |

---

## Dataset

Public road surface image dataset from Kaggle containing images across multiple
road surface conditions. Organized into 3 classes matching the paper:
**Dry, Wet, and Muddy**.

Dataset: `hemanthhari/road-surface-dataset`  
Total images: 10,680  
Classes: Dry (3,660) | Wet (3,560) | Muddy (3,460)
