#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATStaticDetection, nets 
from NEATUtils import helpers

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[2]:


#Specify the location and name of the npz file for training and validation data
NpzDirectory = '/home/sancere/VarunNewton/CurieTrainingDatasets/O-NEAT/'
TrainModelName = 'StaticONEATBin2.npz'
ValidationModelName = 'StaticONEATBin2Validation.npz'

#Read and Write the h5 file, directory location and name
Model_dir = '/home/sancere/VarunNewton/CurieDeepLearningModels/O-NEATweights/'
Model_Name = 'StaticONEXTd29cl48ca16.h5'


# In[3]:


#Specify thre training parameters

#Normal Events = 0, Apoptosis = 1, Division = 2
categories = 6
batch_size = 80

epochs = 150
depth = 29
cardinality = 16

show = False
TrainModel = nets.resnext_v2
model_weights = Model_dir + Model_Name

if os.path.exists(model_weights):

    model_weights = None
    print('loading weights')
else:
   
    model_weights = None

Categories_Name = []
Categories_Name.append(['Normal', 0])
Categories_Name.append(['Apoptosis', 4])
Categories_Name.append(['Divisions', 5])
Categories_Name.append(['MacroKitty', 1])
Categories_Name.append(['NonMatureP1', 2])
Categories_Name.append(['MatureP1', 3])


# In[ ]:


global Trainingmodel
#Initate training of the model
Trainingmodel = NEATStaticDetection(NpzDirectory, TrainModelName,ValidationModelName, categories, Categories_Name, Model_dir, Model_Name, TrainModel, depth = depth, model_weights = model_weights, cardinality = cardinality ,epochs = epochs, batch_size = batch_size, show = show )


# In[ ]:





# In[ ]:





# In[ ]:




