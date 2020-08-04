# Wafer map pattern classification

## 0. Data Description

<img src="https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/Data%20description.jpg?raw=true" alt="Data description.jpg" width="60%" height="60%" >




## 1. Manual Feature Extraction approaches

![MFE.png](https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/MFE.png?raw=true)

### 1] Feature Extraction

#### 1) Density features

![alt-text-1](https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/density_features1.png?raw=true) ![alt-text-2](https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/density_features2.png?raw=true)

* a wafer map is divided into 13 regions (4 edge regions, 9 central regions)
* defect densities of each of regions are used as density features
* 13 extracted features 

#### 2) Geometry features

<img src="https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/geometry.png?raw=true" alt="geometry.png" width="60%" height="60%">

* a salient area is extracted through noise filtering
* based on the salient region with maximum area, six geometric features are extracted
  * perimeter, area, length of minor axes, length of major axes, solidity and eccentricity
* 6 extracted features 

#### 3) Radon features

![radon_features.png](https://github.com/DMkelllog/dmkelllog.github.io/blob/master/assets/images/radon_features.png?raw=true)

* created by radon transformation
  * creates two-dimensional representation of a wafer map based on a series of projections
* cubic interpolation is applied to obtain the same number of rows.
* 20 rows from the result of radon transformation and extracted row mean <img src="https://render.githubusercontent.com/render/math?math=R_\mu"> and row standard deviation <img src="https://render.githubusercontent.com/render/math?math=R_\sigma"> for each row
* 40 extracted features 

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

#### 3) Training settings

* batch size: 64
* optimizer: adam
* learning rate: 10<sup>-4</sup>
* max epoch: 1000
* early stopping: 10 consecutive stagnation of validation loss

## 3) Evaluation

#### 1) Data split

* Train size
  * 500, 5000, 50000, 162946 (all available)
  * 20% of train size used for validation
* Test size
  * 10000

#### 2) Metrics

* macro average F1 score
* micro average F1 score

#### 3) Replicaitons

* 10 replications
  * different data split
  * differemt random seed
* mean and standard deviation
