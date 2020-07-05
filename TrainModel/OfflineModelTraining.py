#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATDetection, nets 
from NEATUtils import helpers

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[2]:


#Specify the location and name of the npz file for training and validation data
NpzDirectory = '/home/sancere/VarunNewton/CurieTrainingDatasets/O-NEAT/'
TrainModelName = 'ONEATBinA.npz'
ValidationModelName = 'ONEATBinAValidation.npz'

#Read and Write the h5 file, directory location and name
Model_dir = '/home/sancere/VarunNewton/CurieDeepLearningModels/O-NEATweights/'
Model_Name = 'OSNETd29K3.h5'
Copy_Model_Name = 'ORNETd29cl48ls16.h5'

# In[3]:


#Specify thre training parameters

#Normal Events = 0, Apoptosis = 1, Division = 2
categories = 6
batch_size = 15
lstm_hidden_units = 16
epochs = 150
depth = 29
cardinality = 16

show = False
TrainModel = nets.OSNET
model_weights = Model_dir + Copy_Model_Name

if os.path.exists(model_weights):

    model_weights = model_weights
    print('loading weights')
else:
   
    model_weights = None

Categories_Name = []
Categories_Name.append(['Normal', 0])
Categories_Name.append(['Apoptosis', 1])
Categories_Name.append(['Divisions', 2])
Categories_Name.append(['MacroKitty', 3])
Categories_Name.append(['NonMatureP1', 4])
Categories_Name.append(['MatureP1', 5])


# In[ ]:


global Trainingmodel
#Initate training of the model
Trainingmodel = NEATDetection(NpzDirectory, TrainModelName,ValidationModelName, categories, Categories_Name, Model_dir, Model_Name, TrainModel, depth = depth, model_weights = model_weights, cardinality = cardinality, lstm_hidden_unit1 = lstm_hidden_units
             ,epochs = epochs, batch_size = batch_size, show = show )


# In[ ]:





# In[ ]:





# In[ ]:




