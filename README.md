# Brain-tumor-and-Alzheimer's-Detection using CNNs

A Convolutional Neural Network (CNN) model to detect and classify brain tumor grades using MR Spectroscopic Imaging (MRSI) data.

## Overview
This project focuses on using CNNs to predict the grades of brain tumors by analyzing MR Spectroscopic Imaging (MRSI) data, which provides biochemical information about brain tissues. The model was trained to distinguish between tumorous and non-tumorous brain tissues and classify tumor grades, aiding in early diagnosis and treatment planning. The study addressed challenges like limited labeled data by using pre-trained models, data augmentation, and careful preprocessing.

## Dataset
- Source: Kaggle  
- Size: 3,533 MRI scans (2,934 tumorous, 599 non-tumorous)  
- Format: 3D MRI scans converted to 2D patches  
- Preprocessing:  
  - Resized images to 128x128x3 for compatibility with ResNet50, InceptionV3, and Xception models.  
  - Normalized pixel values (0-255 to 0-1).  
  - Removed unlabeled images.  
  - Noted class imbalance (more tumorous images) as a challenge.

## Methodology
- **Data Splitting**: Used Stratified K-Fold Cross Validation (5 folds). Each fold had 1,173 images, with 4,692 images for training and 1,173 for validation per iteration.  
- **Model Training**:  
  - Pre-trained Models: ResNet50, InceptionV3, Xception (frozen feature extraction layers, added problem-specific layers: global average pooling, 256-node dense layer, 1-node output).  
  - Custom S-CNN: 256-filter convolutional layer, ReLU activation, 2x2 max pooling, 128-node dense layer, sigmoid output for binary classification.  
  - Training Parameters: Adam optimizer, learning rate 0.0001, batch size 32, 50 epochs.  
- **Evaluation Metrics**: Accuracy, precision, recall, F1 score, AUC-ROC (using a confusion matrix).

## Implementation
- **Dataset Creation**: Collected and categorized MRI scans of brain tumors.  
- **Training**: Converted images to grayscale, transformed them into arrays using NumPy, and trained CNNs with TensorFlow and OpenCV.  
- **Detection**: The model compares new MRI scans to existing data to classify tumor grades.

## Tools
- Python  
- TensorFlow/Keras  
- NumPy, OpenCV  
- Kaggle (for dataset)

## Skills
- Machine Learning  
- Deep Learning  
- Image Processing  
- Data Preprocessing  
- Transfer Learning

## Results
The CNN models demonstrated potential as a reliable tool for classifying brain tumor grades, with performance evaluated using accuracy, precision, recall, F1 score, and AUC-ROC. The use of pre-trained models and data augmentation helped overcome the challenge of limited labeled data.

## Note
Project files are currently unavailable but will be added soon. This project was part of a collaborative effort at Vignan's Institute of Information Technology, Visakhapatnam, under the guidance of Mrs. Ch. Swapna Priya(Ph.D).
