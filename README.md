# Wafer map pattern classification

## 0. Data Description

![Data description.jpg](https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/Data%20description.jpg?raw=true)

## 1. Manual Feature Extraction approaches

![MFE.png](https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/MFE.png?raw=true)

### 1] Feature Extraction

#### 1) Density features

![alt-text-1](https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/density_features1.png?raw=true) ![alt-text-2](https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/density_features2.png?raw=true)

13 features

#### 2) Geometry features

<img src="https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/geometry.png?raw=true" alt="geometry.png" style="zoom:30%;" />

6 features

#### 3) Radon features

![radon_features.png](https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/radon_features.png?raw=true)

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

![CNN.png](https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/CNN.png?raw=true)

#### 1) Resizing

Scikit-image resize module 

* 'anti-aliasing' disabled (prevent black resizing)
* preserve_range = True

resize size = 32, 64, 96, 128

<img src="https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/resize.png?raw=true" alt="resize.png" style="zoom:67%;" />

#### 2) CNN architecture  

<img src="https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/cnn%20architecture.PNG?raw=true" alt="cnn architecture.PNG" style="zoom: 67%;" />

[images] CNN architecture
