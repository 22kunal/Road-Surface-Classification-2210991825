# Road Surface Classification
### Modern Approaches to Road Surface Classification: Assessing CNN vs. KNN Models

**Student:** Kunal Kumar  
**Roll No:** 2210991825  
**Type:** Research Paper  
**Supervisor:** Dr. Lalit K Sharma  
**Department:** Computer Science and Engineering, Chitkara University  

---

## Project Overview

A hybrid CNN-KNN model for classifying road surfaces into three categories:
**Dry, Wet, and Muddy** using deep learning and instance-based classification.

Achieved **98.31% accuracy** on 10,680 images, outperforming LSTM, Random Forest,
ResNet50, and SVM baselines.

---

## Run the Code Directly in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/22kunal/Road-Surface-Classification-2210991825/blob/main/Source_Code/road_classification_2210991825.ipynb)

> Click the button above to open and run the notebook directly in your browser.
> No installation required. Use Runtime → Run all to execute.

---

## Repository Structure

| Folder | Contents |
|--------|----------|
| `Report_and_PPT/` | Project report (Word) and presentation (PPT) |
| `Source_Code/` | Google Colab notebook — full CNN-KNN implementation |

---

## Results Summary

| Model | Accuracy |
|-------|----------|
| **CNN-KNN Hybrid (Proposed)** | **98.31%** |
| Random Forest | 93.44% |
| ResNet50 | 91.06% |
| LSTM | 90.56% |
| SVM | 89.58% |

---

## How to Run

1. Click the **Open in Colab** button above
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Click **Runtime → Run all**
4. Wait ~20 minutes for training to complete
5. Results, graphs, and confusion matrix will be saved to the `results/` folder

---

## Technologies Used

- Python 3.10
- TensorFlow 2.x / Keras
- Scikit-learn
- Matplotlib / Seaborn
- Google Colab (T4 GPU)
- Kaggle Dataset
