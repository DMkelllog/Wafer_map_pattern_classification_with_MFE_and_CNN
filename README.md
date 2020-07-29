# Wafer map pattern classification

## 0. Data Description

## 1. Manual Feature Extraction approaches

### 1] Feature Extraction
#### 1) Density features
13 features
#### 2) Geometry features
6 features
#### 3) Radon features
40 features

total 59 manually extracted features

### 2] Classifier

#### 1) Random Forest
n_estimator = 100  
(no hyperparameter search)

#### 2) Feed-forward Neural Network
Two dense layers (128 hidden nodes) with relu activation  
** neural network training settings **  
(20% of training dataset used as validation dataset for hyperparameter search)  
(standard scaling with training dataset)

#### 3) Support Vector Machine
RBF kernel, C = 1000, gamma = 0.001  
(20% of training dataset used for hyperparameter search)
(standard scaling with training dataset)

## 2. Convolutional Neural Network approaches
#### 1) Resizing
resize size = 32, 64, 96, 128  
#### 2) CNN architecture  
[images] CNN architecture
