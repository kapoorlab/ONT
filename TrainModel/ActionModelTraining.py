#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


TrainDataDir = '/data/u934/service_imagerie/v_kapoor/CSVforNeat/YolONEAT/CenterTrainData/'
Model_dir = '/data/u934/service_imagerie/CurieTrainingDatasets/YolONEAT/CurieDeepLearningModels/YolONEAT/'
Model_Name = 'ORYolo.h5'


# In[ ]:


#Neural network parameters
#For ORNET use residual = True and for OSNET use residual = False
residual = True
#NUmber of starting convolutional filters, is doubled down with increasing depth
startfilter = 48
#CNN network start layer, mid layers and lstm layer kernel size
start_kernel = 3
lstm_kernel = 3
mid_kernel = 3
#Network depth has to be 9n + 2, n= 3 or 4 is optimal for Notum dataset
depth = 29
#Training epochs, longer the better zith proper chosen leqrning rate
epochs = 150
#Size of the gradient descent length vector, start small and use callbacs to get smaller when reaching the minima
learning_rate = 1.0E-4
#For stochastic gradient decent, the batch size used for computing the gradients
batch_size = 10
#Number of LSTM hidden layers > time sequence used for training
lstm = 16
# use softmax for single event per box, sigmoid for multi event per box
last_activation = 'softmax'
#X Y T H W Confidence Angle makes up the box vector
box_vector = 7
# Grid and anchor boxes for yolo
gridX = 1
gridY = 1
anchors = 1
lambdacord = 1.5


# In[ ]:


config = NeatConfig(residual = residual, startfilter = startfilter, start_kernel = start_kernel, 
                    mid_kernel = mid_kernel,lstm_kernel = lstm_kernel,
                    depth = depth, lstm = lstm, learning_rate = learning_rate, batch_size = batch_size,
                    epochs = epochs, ModelName = Model_Name,
                    gridX = gridX, gridY = gridY, anchors = anchors, lambdacord = lambdacord, last_activation = last_activation)

config_json = config.to_json()
show = False

model_weights = Model_dir + Model_Name

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
print(config)
save_json(config_json, Model_dir + Model_Name + 'Param.json')


# In[ ]:


Train = NEATDetection(config, TrainDataDir, Categories_Name, box_vector, Model_dir, Model_Name, model_weights = model_weights, show = show)

Train.loadData()

Train.TrainModel()


# In[ ]:





# In[ ]:





# In[ ]:




