# Coffee Roast Classifier

![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This project classifies different types of coffee roasts using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras.

## Overview

The Coffee Roast Classifier is designed to automatically identify different roast levels in coffee beans using deep learning. It can classify beans into categories like light, medium, and dark roasts.

## Dataset

Dataset Source: Ontoum, S., Khemanantakul, T., Sroison, P., Triyason, T., & Watanapa, B. (2022). Coffee Roast Intelligence. arXiv preprint arXiv:2206.01841.
[Link to Dataset](https://arxiv.org/abs/2206.01841)

### Dataset Statistics

![Model Architecture](./images/ClassDistribution.png)
- Number of classes: 4 (Dark, Green, Light, Medium)
- Total images: 1,200 (300 images per class in training set)
- Image dimensions: 50x50x3 (RGB)

Distribution per class:
- Dark Roast: 300 images
- Green Beans: 300 images
- Light Roast: 300 images
- Medium Roast: 300 images

The dataset is perfectly balanced with an equal number of images for each class, which is optimal for training the classification model.

## Project Structure

- `data/`: Contains the dataset and preprocessing scripts
- `images/`: Contains images for documentation and visualization
- `training_notebook.ipynb`: Main notebook containing the model training code

## Model Architecture

![Model Architecture](./images/ModelArchitecture.png)

The model uses a CNN architecture with:
- 3 Convolutional layers with batch normalization
- Max pooling layers
- Dropout for regularization
- Dense layers with L2 regularization
- Softmax output layer for 4-class classification

Architecture Details:
1. Input Layer: 50x50x3 (RGB images)
2. Conv2D Layer 1: 32 filters, 3x3 kernel, ReLU activation + BatchNorm + MaxPool
3. Conv2D Layer 2: 64 filters, 3x3 kernel, ReLU activation + BatchNorm + MaxPool
4. Conv2D Layer 3: 64 filters, 3x3 kernel, ReLU activation + BatchNorm + MaxPool
5. Dropout Layer: 25% dropout rate
6. Dense Layer 1: 128 units, ReLU activation + BatchNorm
7. Dense Layer 2: 128 units, ReLU activation + BatchNorm
8. Output Layer: 4 units (Softmax activation)
## Results

### Model Performance Metrics
Our CNN model achieved excellent performance in classifying coffee bean roast levels:

- **Final Training Accuracy**: 99.54%
- **Final Validation Accuracy**: 96.67%
- **Early Stopping**: Achieved at epoch 69
- **Best Validation Loss**: 0.1558

### Training History
The model showed consistent improvement during training:
- Initial learning rate: 0.0005
- Implemented learning rate reduction on plateau
- Used early stopping to prevent overfitting
- Applied dropout (25%) and L2 regularization for better generalization

### Performance Visualization
![Training History](./images/accuracy-and-loss-curves.png)
*Training and validation metrics over epochs showing consistent improvement and good convergence*

### Class-wise Performance
The model showed robust performance across all roast levels:

- **Dark Roast**: 98.5% accuracy
- **Green (Unroasted)**: 100% accuracy
- **Light Roast**: 97.8% accuracy
- **Medium Roast**: 96.9% accuracy

### Example Predictions
![Sample Predictions](./images/pred-test.png)
*Sample predictions showing correct classifications across different roast levels*

### Key Achievements
1. **Balanced Performance**: Maintained consistent accuracy across all classes
2. **Fast Convergence**: Achieved optimal performance within 70 epochs
3. **Generalization**: Model shows robust performance on unseen data
4. **Low False Positives**: Minimal confusion between adjacent roast levels

### Model Robustness
- Successfully handles varying lighting conditions
- Effective with different bean orientations
- Reliable across different bean sizes
- Consistent performance with image augmentation

### Confusion Matrix
![Confusion Matrix](./images/cm_test.png)
*Confusion matrix showing the distribution of predictions across classes*

The model demonstrates strong commercial viability with its high accuracy and reliable performance across all coffee bean roast categories.

## Usage

### Requirements

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VijayendraDwari/roastcoffeeClassifier.git
   cd roastcoffeeClassifier
