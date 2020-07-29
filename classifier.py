#!/usr/bin/env python
# coding: utf-8

# In[1]:


from preprocess import import_wm_data, split_data, standard_scale
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import CNN_base, FNN, RF, SVM
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


# In[ ]:


class Classifier():
    
    def __init__(self):
        self.es = EarlyStopping(monitor='val_loss',patience=10, mode='auto', restore_best_weights=True)
        self.BATCH_SIZE = 256 * 4
        self.MAX_EPOCH = 1000
    
    def MFE(self, X_split, y_split, model):
        if model == 'SVM':
            X_split_scaled = standard_scale(X_split)
            Model = SVM()
            Model.fit(X_split_scaled[0], y_split[0])
            y_hat = Model.predict(X_split_scaled[2])
        
        elif model == 'RF':
            Model = RF()
            Model.fit(np.concatenate([X_split[0],X_split[1]]), np.concatenate([y_split[0],y_split[1]]))
            y_hat = Model.predict(X_split[2])
        
        elif model == 'FNN':
            X_split_scaled = standard_scale(X_split)
            Model = FNN(model)
            Model.fit(X_split_scaled[0], y_split[0],
                    validation_data=[X_split_scaled[1], y_split[1]],
                    epochs=self.MAX_EPOCH,
                    batch_size=self.BATCH_SIZE,
                    callbacks=[self.es])
            y_hat = Model.predict_classes(X_split_scaled[2])
        
        else:
            print('model undefined')
        
        return self.evaluate(y_split[2], y_hat)
    
    def CNN(self, X_split, y_split, dim):
        Model = CNN_base(dim)
        Model.fit(X_split[0], y_split[0],
                    validation_data=[X_split[1], y_split[1]],
                    epochs=self.MAX_EPOCH,
                    batch_size=self.BATCH_SIZE,
                    callbacks=[self.es])
        y_hat = Model.predict_classes(X_split[2])
        return self.evaluate(y_split[2], y_hat)
        
            
    def evaluate(self, y_true, y_hat):
        macro = f1_score(y_true, y_hat, average='macro')
        micro = f1_score(y_true, y_hat, average='micro')
        cm = (confusion_matrix(y_true, y_hat))
        print('\n\nmacro: {}, micro: {}\n\n' .format(macro, micro))
        return macro, micro, cm
        
    #####################################################
    def classifier(self, rep_id, method, model):
        print('\n\n\nrep:     {}\nmethod:  {}\nmodel:   {}\n\n' .format(str(rep_id+1), method, model))
        # data import
        X, y = import_wm_data(mode=method, dim=model)
        # data split
        X_split, y_split = split_data(X,y, RAND_NUM=rep_id)
        
        # build each model 
        if method == 'MFE':
            return self.MFE(X_split, y_split, model)
        if method == 'CNN':
            return self.CNN(X_split, y_split, model)
    #####################################################

