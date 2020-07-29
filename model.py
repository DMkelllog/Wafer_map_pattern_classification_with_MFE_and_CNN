#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.optimizers import Adam


# In[2]:


def RF():
    model = RandomForestClassifier(n_estimators=100)
    return model

def SVM():
    model = SVC(C = 1000, gamma=0.001)
    return model

def CNN_base(dim):
    model = Sequential([
        Input([dim,dim,1]),
        Conv2D(16, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(32, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(256, 3, activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(9, activation='softmax')
    ])
    model.compile(optimizer= Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def FNN(dim):
    model = Sequential([
        Input([59,]),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(9, activation='softmax')
    ])
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# In[ ]:




