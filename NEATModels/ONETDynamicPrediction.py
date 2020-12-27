#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:56:35 2020

@author: aimachine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 12:01:40 2019

@author: vkapoor
"""


import numpy as np
import tensorflow as tf
#from keras.utils import to_categorical
from keras.utils import multi_gpu_model
from NEATUtils.helpers import  Yoloprediction,zero_pad
from NEATUtils.helpers import  chunk_list
from NEATUtils.helpers import normalizeFloatZeroOne


class ONETDynamicPrediction(object):
    
    
   def __init__(self, image, ModelA, ModelB, ModelAConfig, ModelBConfig, inputtime, KeyCategories, KeyCord, stride = 4, n_tiles = 1, 
                overlap_percent = 0.8 ):
        
        self.image = image
        self.ModelA = ModelA
        self.ModelB = ModelB
        self.ModelAconfig = ModelAConfig
        self.ModelBconfig = ModelBConfig
        self.KeyCategories = KeyCategories
        self.KeyCord = KeyCord
        self.stride = stride
        self.inputtime = inputtime
        self.n_tiles = n_tiles
        self.originalX = self.image.shape[1]
        self.originalY = self.image.shape[2]
        self.n_tiles = n_tiles
        self.overlap_percent = overlap_percent
        self.TrainshapeX = ModelAConfig["sizeX"]
        self.TrainshapeY = ModelAConfig["sizeY"]
        self.sizeTminus = ModelAConfig["sizeTminus"]
        self.sizeTplus = ModelAConfig["sizeTplus"]
        self.TimeFrames = self.Tminus + self.sizeTplus

   def GetTiles(self):
       
       return self.n_tiles
        
   def GetLocationMaps(self):    
                
                 
                 EventBoxes = []
                 self.image = normalizeFloatZeroOne(self.image,1,99.8)          
                 #Break image into tiles if neccessary
                 AllPredictions, AllX, AllY = self.PredictMemory(self.image)
                 #Iterate over tiles
                 for p in range(0,len(AllPredictions)):   

                   sum_time_prediction = AllPredictions[p]
                   
                   if sum_time_prediction is not None:
                   
                    for i in range(0, sum_time_prediction.shape[0]):
                         
                         time_prediction =  sum_time_prediction[i]
                         #This method returns a dictionary of event names and output vectors and we collect it for all tiles as a list
                         EventBoxes = EventBoxes + Yoloprediction(self.image, AllY[p], AllX[p], time_prediction, self.stride, self.inputtime, self.KeyCategories, self.KeyCord, self.TrainshapeX, self.TrainshapeY, self.TimeFrames, self.Mode, 'Dynamic')
                 
                 self.EventBoxes =  EventBoxes
               
        
   def OverlapTiles(self):
        
    if self.n_tiles == 1:
        
               patchshape = (self.image.shape[1], self.image.shape[2])  
              
               image = zero_pad(self.image, self.stride,self.stride)

               patch = []
               rowout = []
               column = []
               
               patch.append(image)
               rowout.append(0)
               column.append(0)
             
    else:
          
     PatchX = self.image.shape[2] // self.n_tiles
     PatchY = self.image.shape[1] // self.n_tiles

     if PatchX > self.TrainshapeX and PatchY > self.TrainshapeY:
      if self.overlap_percent > 1 or self.overlap_percent < 0:
         self.overlap_percent = 0.8
     
      jumpX = int(self.overlap_percent * PatchX)
      jumpY = int(self.overlap_percent * PatchY)
     
      patchshape = (PatchY, PatchX)   
      rowstart = 0; colstart = 0
      Pairs = []  
      #row is y, col is x
      
      while rowstart < self.image.shape[1] - PatchY:
         colstart = 0
         while colstart < self.image.shape[2] - PatchX:
            
             # Start iterating over the tile with jumps = stride of the fully convolutional network.
             Pairs.append([rowstart, colstart])
             colstart+=jumpX
         rowstart+=jumpY 
        
      #Include the last patch   
      rowstart = self.image.shape[1] - PatchY
      colstart = 0
      while colstart < self.image.shape[2] - PatchX:
                    Pairs.append([rowstart, colstart])
                    colstart+=jumpX
      rowstart = 0
      colstart = self.image.shape[2] - PatchX
      while rowstart < self.image.shape[1] - PatchY:
                    Pairs.append([rowstart, colstart])
                    rowstart+=jumpY              
                    
      if self.image.shape[1] >= self.TrainshapeY and self.image.shape[2]>= self.TrainshapeX :          
          
            patch = []
            rowout = []
            column = []
            for pair in Pairs: 
               smallpatch, smallrowout, smallcolumn =  chunk_list(self.image, patchshape, self.stride, pair)
               patch.append(smallpatch)
               rowout.append(smallrowout)
               column.append(smallcolumn) 
        
     else:
         
               patch = []
               rowout = []
               column = []
               image = zero_pad(self.image, self.stride,self.stride)
               
               patch.append(image)
               rowout.append(0)
               column.append(0)
               
    self.patch = patch          
    self.sY = rowout
    self.sX = column   

  
   def PredictMemory(self,sliceregion):
            try:
                self.OverlapTiles()
                AllPredictions = []
                AllX = []
                AllY = []
                for i in range(0,len(self.patch)):   
                   
                   sum_time_prediction = self.MakePatches(self.patch[i])
                   #print('Applying prediction in patch starting at (XY):', self.sX[i], self.sY[i] )
                   #print('Ending patch prediction location (XY):', self.sX[i] +  self.image.shape[2] // self.n_tiles, self.sY[i] +  self.image.shape[1] // self.n_tiles)

                   AllPredictions.append(sum_time_prediction)
                  
                   AllX.append(self.sX[i])
                   AllY.append(self.sY[i])
           
            except tf.errors.ResourceExhaustedError:
                
                print('Out of memory, increasing overlapping tiles for prediction')
                
                self.n_tiles = self.n_tiles  + 1
                
                print('Tiles: ', self.n_tiles)
                
                self.PredictMemory(sliceregion)
                
            return AllPredictions, AllX, AllY
        
   def MakePatches(self, sliceregion):
       
       prediction_vectorDynamic = None
       
       prediction_vectorDynamicA = None
       
       predict_im = np.expand_dims(sliceregion,0)
       
       #Apply model prediction using multi gpu
       if self.ModelA is not None:
          try: 
             parallel_modelA = multi_gpu_model(self.ModelA, gpus=None)
          except:
             parallel_modelA = self.ModelA 
          prediction_vectorDynamicA = parallel_modelA.predict(np.expand_dims(predict_im,-1), verbose = 0)
       if self.ModelB is not None: 
          try: 
              parallel_modelB = multi_gpu_model(self.ModelB, gpus=None)
          except:
              parallel_modelB = self.ModelB
          prediction_vectorDynamic = parallel_modelB.predict(np.expand_dims(predict_im,-1), verbose = 0)
       else:
          prediction_vectorDynamic = prediction_vectorDynamicA 
         
       for i, bi in enumerate(prediction_vectorDynamicA): prediction_vectorDynamic[i] += bi
       for i, bi in enumerate(prediction_vectorDynamic): prediction_vectorDynamic[i] /= 2 
            
       return prediction_vectorDynamic      
         
  
 
    
    
    
