# Tom & Jerry Image Classification

This repository is designed for analysis and training a deep learning model on a Tom and Jerry dataset.

## Dataset

The dataset was kindly provided by a Kaggle user: [[Kaggle Dataset Link Here](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification/data)].

---

## Repository Structure

* `eda.ipynb`: Contains the Exploratory Data Analysis (EDA) of the dataset.
* `model_layers.py`: Contains the definition and initialization of the PyTorch neural network model (`LightweightGoogLeNet`).
* `helper_functions.py`: Contains the training and testing/evaluation loops for the model.
* `main.ipynb`: Demonstrates model initialization, training, testing, and performance evaluation.

---

## Model Architecture

The model architecture (`LightweightGoogLeNet`) is inspired by **GoogLeNet (Inception v1)**, adapted into a lightweight variant suitable for efficient training while maintaining high visual recognition capabilities.

| Type / Layer | Patch Size / Stride | Input Size | Output Channels | #1x1 | #3x3 reduce | #3x3 | #5x5 reduce | #5x5 | Pool proj |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Convolution** | 7x7 / 2 | 224x224x3 | 32 | - | - | - | - | - | - |
| **Max Pooling** | 3x3 / 2 | 112x112x32 | 32 | - | - | - | - | - | - |
| **Convolution** | 1x1 / 1 | 56x56x32 | 32 | - | - | - | - | - | - |
| **Convolution** | 3x3 / 1 | 56x56x32 | 64 | - | - | - | - | - | - |
| **Max Pooling** | 3x3 / 2 | 56x56x64 | 64 | - | - | - | - | - | - |
| **Inception 3a** | - | 28x28x64 | 128 | 32 | 48 | 64 | 8 | 16 | 16 |
| **Inception 3b** | - | 28x28x128 | 208 | 64 | 64 | 96 | 8 | 16 | 32 |
| **Max Pooling** | 3x3 / 2 | 28x28x208 | 208 | - | - | - | - | - | - |
| **Inception 4a** | - | 14x14x208 | 256 | 96 | 48 | 104 | 8 | 24 | 32 |
| **Inception 4b** | - | 14x14x256 | 256 | 80 | 56 | 112 | 12 | 32 | 32 |
| **Inception 4c** | - | 14x14x256 | 256 | 64 | 64 | 128 | 12 | 32 | 32 |
| **Inception 4d** | - | 14x14x256 | 264 | 56 | 64 | 144 | 16 | 32 | 32 |
| **Inception 4e** | - | 14x14x264 | 400 | 128 | 80 | 160 | 16 | 48 | 64 |
| **Max Pooling** | 3x3 / 2 | 14x14x400 | 400 | - | - | - | - | - | - |
| **Inception 5a** | - | 7x7x400 | 400 | 128 | 80 | 160 | 16 | 48 | 64 |
| **Inception 5b** | - | 7x7x400 | 480 | 192 | 96 | 192 | 16 | 48 | 48 |
| **Avg Pooling** | 7x7 / 1 | 7x7x480 | 480 | - | - | - | - | - | - |
| **Dropout (40%)** | - | 1x1x480 | 480 | - | - | - | - | - | - |
| **Linear FC** | - | 480 | 4 | - | - | - | - | - | - |

---

## Results

* **Test Accuracy:** The trained model achieved **94% accuracy** on the test dataset.
