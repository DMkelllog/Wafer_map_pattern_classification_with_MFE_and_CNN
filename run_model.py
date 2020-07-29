#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
import sys
from classifier import Classifier


# In[2]:


rep_id = int(sys.argv[1])
method_id = int(sys.argv[2])
MFE_model_id = int(sys.argv[3])
CNN_model_id = int(sys.argv[4])


# In[3]:


method_list = ['MFE', 'CNN']
model_list = [['RF', 'FNN', 'SVM'], [32, 64, 96, 128]]


# In[ ]:


method = method_list[method_id]

if method == 'MFE':
    model = model_list[method_id][MFE_model_id]
if method == 'CNN':
    model = model_list[method_id][CNN_model_id]
    
clf = Classifier()
macro, micro, cm = clf.classifier(rep_id, method, model)

filename_f1 = './result/{0}_{1}_{2}_{3}.csv'.format(rep_id, method, model, 'f1')
filename_cm = './result/{0}_{1}_{2}_{3}.csv'.format(rep_id, method, model, 'cm')

with open(filename_f1, 'w') as f:
    w = csv.writer(f)
    w.writerow([macro, micro])
    
with open(filename_cm, 'w') as f:
    w = csv.writer(f)
    w.writerows(cm)

