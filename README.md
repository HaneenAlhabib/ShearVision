# 🐑 ShearVision: Local Sheep Breed Classification

ShearVision is an automated computer vision-based system for classifying local sheep breed images using DINOv2 Vision Transformer transfer learning.

## Project Overview

The model classifies sheep images into seven classes:

- Naeimi
- Najdi
- Harri
- Sawakni
- Roman
- Barbari
- Goat

The system uses the Kaggle Eid Al-Adha 2025 Sheep Classification Challenge dataset after cleaning and label correction.

## Methodology

The ShearVision pipeline includes:

1. Image loading
2. Dataset cleaning and label correction
3. Duplicate/corrupt image removal
4. Image resizing to 518 × 518
5. Data augmentation
6. ImageNet normalization
7. DINOv2 Vision Transformer feature extraction
8. Custom seven-class classification head
9. Breed prediction

## Results

| Evaluation | Result |
|---|---:|
| Held-out test accuracy | 99.01% |
| Macro-F1 | 99.06% |
| Weighted-F1 | 99.03% |
| Best validation accuracy | 98.25% |
| Strict 5-fold CV accuracy | 97.17% ± 1.73% |
| Strict 5-fold CV macro-F1 | 96.29% ± 2.33% |

## Dataset

The dataset used in this project is from the Kaggle Eid Al-Adha 2025 Sheep Classification Challenge.

Dataset link:  
https://www.kaggle.com/competitions/sheep-classification-challenge-2025

The dataset is not included in this repository. Please download it from Kaggle and update the dataset path in the notebook before running.

## How to Run

1. Download the dataset from Kaggle.
2. Open `ShearVision_Final.ipynb` in Google Colab.
3. Update the dataset path if needed.
4. Install the required libraries.
5. Run all notebook cells.

## Computational Environment

Experiments were conducted in Google Colab using an NVIDIA Tesla T4 GPU with 14.56 GB GPU memory.

Main libraries:

- Python 3.12.13
- PyTorch 2.10.0+cu128
- CUDA 12.8
- timm 1.0.26
- scikit-learn 1.6.1
- NumPy 2.0.2
- pandas 2.2.2
- Matplotlib 3.10.0
- OpenCV 4.13.0
- PIL 11.3.0

