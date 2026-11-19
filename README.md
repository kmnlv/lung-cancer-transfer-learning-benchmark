# Lung Cancer Histopathology Classification with Pretrained CNNs

> Course: Deep Learning · Semester 2

## Overview

Transfer-learning benchmark for classifying lung histopathology images into three classes: **Sano**, **Adenocarcinoma** and **Célula Escamosa**. Three ImageNet-pretrained architectures are fine-tuned under an identical augmentation and evaluation pipeline.

## Repository structure

- `notebooks/lung-cancer-transfer-learning.ipynb` - full experiment with curves, confusion matrix and classification report.

## Method

- `ImageDataGenerator` with rotation, shift, shear, zoom and horizontal/vertical flips; 224 x 224 inputs; 15% validation split.
- Frozen convolutional bases from **DenseNet121**, **ResNet50** and **VGG16** (`include_top=False`, `pooling='avg'`).
- Shared classification head: `Flatten -> Dropout(0.2) -> Dense(256) -> BatchNorm -> Dense(64) -> BatchNorm -> Dense(3, softmax)`.
- Adam (1e-4), 5 epochs; accuracy/loss curves, confusion matrix and per-class metrics.

## Results

The target was at least 96% accuracy. **ResNet50 achieved 98.22%**, outperforming DenseNet121 and VGG16 in the final comparison.

## Data

Kaggle dataset `andrewmvd/lung-and-colon-cancer-histopathological-images` (12,750 training and 2,250 validation images). Images are not stored in this repository because of their size; download them via KaggleHub and update the cached path if necessary.

## Requirements

`tensorflow`, `scikit-learn`, `matplotlib`, `numpy` and `jupyter`.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/lung-cancer-transfer-learning.ipynb
```

---
Author: Karla Maria Monroy Alvarez
