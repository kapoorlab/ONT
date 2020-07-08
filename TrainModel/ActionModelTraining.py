#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATDetection, nets
from NEATModels.config import NeatConfig
from NEATUtils import helpers
from NEATUtils.helpers import save_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[2]:


NpzDirectory = '/home/sancere/VarunNewton/CurieTrainingDatasets/O-NEAT/'
TrainModelName = 'MasterBigONEAT.npz'
ValidationModelName = 'MasterBigONEATValidation.npz'

#Read and Write the h5 file, directory location and name
Model_dir = '/home/sancere/VarunNewton/CurieDeepLearningModels/O-NEATweights/'
Model_Name = 'MidONEATd29cl48.h5'

#Neural network parameters
#For ORNET use residual = True and for OSNET use residual = False
residual = True
startfilter = 48
start_kernel = 7
mid_kernel = 3
depth = 29
epochs = 150
learning_rate = 1.0E-4
batch_size = 10
lstm = 16


# In[4]:


config = NeatConfig(startfilter = startfilter, start_kernel = start_kernel, mid_kernel = mid_kernel,
                depth = depth, lstm = lstm, learning_rate = learning_rate, batch_size = batch_size, epochs = epochs)

config_json = config.to_json()
show = True

model_weights = Model_dir + Model_Name

if os.path.exists(model_weights):

    model_weights = model_weights
    print('loading weights')
else:
   
    model_weights = None

Categories_Name = []
Categories_Name = {
    0:"Normal",
    1:"Apoptosis",
    2:"Divisions",
    3:"MacroCheate",
    4:"NonMatureP1",
    5:"MatureP1"
}
print(config)
save_json(config_json, Model_dir + 'ActionModelParameterFile.json')


# In[ ]:


Train = NEATDetection(config, NpzDirectory, TrainModelName,ValidationModelName, Categories_Name, Model_dir, Model_Name, model_weights = model_weights, show = show)

Train.loadData()

Train.TrainModel()


# In[ ]:





# In[ ]:





# In[ ]:




