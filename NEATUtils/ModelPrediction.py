#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 07:59:00 2020

@author: aimachine
"""
from NEATModels import ONETDynamicPrediction,ONETStaticPrediction, ONETSmartPrediction, ONETSmartSancerePrediction, ONETLivePrediction
from NEATUtils import NMS
from NEATUtils.helpers import MakeTrees, DensityCounter
from tqdm import tqdm
from natsort import natsorted
import numpy as np
import os
from NEATUtils.helpers import MarkerToCSV
import cv2
import h5py
import math
from stardist.models import StarDist2D
from tifffile import imread, imwrite
from keras.models import load_model
from NEATModels import time_yolo_loss, static_yolo_loss, Concat
import csv
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.filters import threshold_otsu
from csbdeep.utils import normalize
from NEATUtils.helpers import smallzero_pad, load_json
from NEATUtils.NMS import drawimage
from scipy.ndimage.morphology import  binary_dilation
from skimage.util import invert as invertimage
from scipy import ndimage as ndi
from skimage import measure
from skimage import morphology
import glob

           





"""
This method applies segmentation model to the input image to generate markers needed by ONEAT to make refined predictions
"""
               
def GenerateMarkers(image, model, n_tiles = 2):
    
    print("Generating Markers", image.shape)
    
    MarkerImage = np.zeros_like(image) 
    for i in tqdm(range(0, image.shape[0])):
            x = image[i,:]
            originalX = x.shape[0]
            originalY = x.shape[1]  
            
            #Get Name
            #Stardist Prediction
            axis_norm = (0,1)
            x = normalize(x,1,99.8,axis=axis_norm)
            x = smallzero_pad(x, 64, 64)
        
            MidImage, details = model.predict_instances(x, n_tiles = (n_tiles, n_tiles))
            StarImage = MidImage[:originalX, :originalY]
            
            properties = measure.regionprops(StarImage, StarImage)
            Coordinates = [prop.centroid for prop in properties]
            
            Coordinates = sorted(Coordinates , key=lambda k: [k[1], k[0]])
            Coordinates.append((0,0))
            Coordinates = np.asarray(Coordinates)
            coordinates_int = np.round(Coordinates).astype(int)
            markers_raw = np.zeros_like(StarImage)  
            markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
            
            markers = morphology.dilation(markers_raw, morphology.disk(2))
            MarkerImage[i,:] = markers
            
    return MarkerImage    


def BinaryDilation(Image, iterations = 1):

    DilatedImage = binary_dilation(Image, iterations = iterations) 
    
    return DilatedImage
    

                                                              

        
def ConvertModel(ModelDirectory, Model):

      f = h5py.File(ModelDirectory + Model + '.h5', 'r+')
      data_p = f.attrs['training_config']
      data_p = data_p.decode().replace("learning_rate","lr").encode()
      f.attrs['training_config'] = data_p
      f.close()      
      
                
def MatlabTester(Movie, CSVFile, ResultFile, ModelDir, ModelA, ModelB, ModelAConfig, ModelBConfig,
             KeyCategories,KeyCord, TestCategory, n_tiles = 1): 
    
    
      
      TrainshapeX = ModelAConfig["sizeX"]
      TrainshapeY = ModelAConfig["sizeY"]
      sizeTminus = ModelAConfig["sizeTminus"]
      sizeTplus = ModelAConfig["sizeTplus"]
      gridX = ModelAConfig["gridX"]
      gridY = ModelAConfig["gridY"]
      multieventA = ModelAConfig["multievent"]
      ModelA = ModelAConfig["ModelName"]
      Mode = ModelAConfig["Mode"]
      categories = ModelAConfig["categories"]
      box_vector = ModelAConfig["box_vector"]
      lambdacord = ModelAConfig["lambdacord"]
      nboxes = ModelAConfig["nboxes"]
      
      multieventB = ModelBConfig["multievent"]
      ModelB = ModelBConfig["ModelName"]
      
      
      
      
      ConvertModel(ModelDir, ModelA)
      ConvertModel(ModelDir, ModelB)
      
      if multieventA == True:
          NEATModelA =  load_model( ModelDir + ModelA + '.h5',  custom_objects={'loss':time_yolo_loss(categories, gridX, gridY, nboxes, box_vector, lambdacord, 'binary'), 'Concat':Concat})
      if multieventA == False:
          NEATModelA =  load_model( ModelDir + ModelA + '.h5',  custom_objects={'loss':time_yolo_loss(categories, gridX, gridY, nboxes, box_vector, lambdacord, 'notbinary'), 'Concat':Concat})  
        
      if multieventB == True:
          NEATModelB =  load_model( ModelDir + ModelB + '.h5',  custom_objects={'loss':time_yolo_loss(categories, gridX, gridY, nboxes, box_vector, lambdacord, 'binary'), 'Concat':Concat})
      if multieventB == False:
          NEATModelB =  load_model( ModelDir + ModelB + '.h5',  custom_objects={'loss':time_yolo_loss(categories, gridX, gridY, nboxes, box_vector, lambdacord, 'notbinary'), 'Concat':Concat})  
      
        
        
         
         
      Categories_Name = []
      CubicModels = [NEATModelA, NEATModelB]
      Categories_Name.append(['Normal', 0])
      Categories_Name.append(['Apoptosis', 1])
      Categories_Name.append(['Divisions', 2])
      Categories_Name.append(['MacroKitty', 3])
      Categories_Name.append(['NonMatureP1', 4])
      Categories_Name.append(['MatureP1', 5])
      
      
      image = imread(Movie)
      print(image.shape)

      
      time, y, x =   np.loadtxt(CSVFile, delimiter = ",", skiprows = 0, unpack=True)  
      Timelist = []
      Ylist = []
      Xlist = []
      Scorelist= []
      Sizelist = []
     
     #os.remove(CSVfile)
      for t in tqdm(range(0, len(time))):
          
          if time[t] > sizeTplus + 1 and math.isnan(x[t])==False and math.isnan(y[t])== False  :
                   
                   crop_Xminus = x[t] - int(TrainshapeX/2)
                   crop_Xplus = x[t]  + int(TrainshapeX/2)
                   crop_Yminus = y[t]  - int(TrainshapeY/2)
                   crop_Yplus = y[t]  + int(TrainshapeY/2)
              
                   region =(slice(int(time[t] - sizeTminus - 1),int(time[t] + sizeTplus )),slice(int(crop_Yminus), int(crop_Yplus)),
                              slice(int(crop_Xminus), int(crop_Xplus)))
                   crop_image = image[region]
                   if(crop_image.shape[0] >= sizeTminus + sizeTplus + 1  and crop_image.shape[1] >= TrainshapeY - 1 and crop_image.shape[2] >= TrainshapeX - 1 ):
                           
                           score, size = MatDynamicEvents(crop_image,time[t], y[t], x[t], CubicModels, classicNEAT,  Categories_Name, Category
                                           ,Mode,n_tiles,TrainshapeX, TrainshapeY, TimeFrames )
                           
                           Timelist.append(time[t])
                           Ylist.append(y[t])
                           Xlist.append(x[t])
                           Scorelist.append(score)
                           Sizelist.append(size)

      Event_Count = np.column_stack([Timelist,Ylist, Xlist, Scorelist, Sizelist]) 
      Event_data = []
      writer = csv.writer(open(ResultFile, "w"))
      for line in Event_Count:
        Event_data.append(line)
      writer.writerows(Event_data)
 

def SmartONEAT(Moviefile, Markerfile, ResultsDirectory, ModelDirectory, ONEATA, ONEATB,  DownsampleFactor, multievent = True, TimeFrames = 7, Mode = 'Detection',categories = 6, TrainshapeX = 54, TrainshapeY = 54, cut = 0.8, sizeTplus = 3, sizeTminus = 3, n_tiles = 2, densityveto = 10): 
    
    
      ConvertModel(ModelDirectory, ONEATA)
      ConvertModel(ModelDirectory, ONEATB)

      if classicNEAT:
                    
         NEATA =  load_model( ModelDirectory + ONEATA + '.h5',  custom_objects={'loss':mid_yolo_loss(categories), 'Concat':Concat})
         NEATB =  load_model( ModelDirectory + ONEATB + '.h5',  custom_objects={'loss':mid_yolo_loss(categories), 'Concat':Concat})
         
         
      if centerNEAT:
          
         NEATA =  load_model( ModelDirectory + ONEATA + '.h5',  custom_objects={'loss':cat_simple_yolo_loss(categories), 'Concat':Concat})
         NEATB =  load_model( ModelDirectory + ONEATB + '.h5',  custom_objects={'loss':cat_simple_yolo_loss(categories), 'Concat':Concat})
         
      if simplecenterNEAT:
          
         NEATA =  load_model( ModelDirectory + ONEATA + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
         NEATB =  load_model( ModelDirectory + ONEATB + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
         
      else:
         NEATA =  load_model( ModelDirectory + ONEATA + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
         NEATB =  load_model( ModelDirectory + ONEATB + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
          

      Categories_Name = []
      
      Categories_Name.append(['Normal', 0])
      Categories_Name.append(['Apoptosis', 1])
      Categories_Name.append(['Divisions', 2])
      Categories_Name.append(['MacroKitty', 3])
      Categories_Name.append(['NonMatureP1', 4])
      Categories_Name.append(['MatureP1', 5])
      
      fname = os.path.basename(os.path.splitext(Moviefile)[0])
      image = imread(Moviefile)
      MarkerImage = imread(Markerfile)
      originalimage = image
      
      scale_percent = int(100/DownsampleFactor) # percent of original size
      width = int(image.shape[2] * scale_percent / 100)
      height = int(image.shape[1] * scale_percent / 100)
      dim = (width, height)
      # resize image
      print('Sampling the image')
      smallimage = np.zeros([image.shape[0], height, width])
      for i in tqdm(range(0, image.shape[0])):
         smallimage[i,:] = cv2.resize(image[i,:], dim)
    
      image = smallimage 
      
      print('Creating Dictionary of Marker Image')
        
      AllTrees = MakeTrees(MarkerImage)
      
      print('Choosing Auto sampling factor for each region')  
      
      AllDensity = DensityCounter(MarkerImage, TrainshapeX, TrainshapeY)
      
      MegaDivisionBoxes = []
      MegaApoptosisBoxes = []
      
      MasterApop = []
      MasterDiv = []
      
      if classicNEAT:
          
         print('Running conventional ONEAT')
         
      else:
          
         print('Running center ONEAT') 
      for tlocation in tqdm(range(0, image.shape[0]- 1)):
           
           smallimg = CreateVolume(image, TimeFrames, tlocation,TrainshapeX, TrainshapeY,  Mode)  
           smalloriginalimg =  CreateVolume(originalimage, TimeFrames, tlocation,TrainshapeX, TrainshapeY,  Mode)           
           smallmarkerimg =  CreateVolume(MarkerImage, TimeFrames, tlocation,TrainshapeX, TrainshapeY,  Mode) 
         
           if smallimg.shape[0]==TimeFrames:
               
               PredictionEvents =  ONETSmartPrediction(smalloriginalimg, smallimg, smallmarkerimg, AllTrees, AllDensity, NEATA, NEATB, DownsampleFactor, tlocation,
                                                    Categories_Name, TrainshapeX, TrainshapeY,sizeTplus, sizeTminus, TimeFrames, Mode, classic = classicNEAT, cut = cut, n_tiles = n_tiles, overlap_percent = 0.8 )
           
               PredictionEvents.GetLocationMaps()
               n_tiles = PredictionEvents.GetTiles()  
               LocationBoxesDivision = PredictionEvents.LocationBoxesDivision
               LocationBoxesApoptosis = PredictionEvents.LocationBoxesApoptosis
               
               
               if LocationBoxesApoptosis is not None: 
                   for j in range(0, len(LocationBoxesApoptosis)):
            
                     boxes, Label = LocationBoxesApoptosis[j]
                     MegaApoptosisBoxes.append(boxes)
                    
               if LocationBoxesDivision is not None:
                   for j in range(0, len(LocationBoxesDivision)):

                     boxes, Label = LocationBoxesDivision[j]
                     MegaDivisionBoxes.append(boxes)        
        
               if tlocation%TimeFrames == 0:
                       MasterApop, MegaApoptosisBoxes =  NMS.StaticEvents(MegaApoptosisBoxes, DownsampleFactor)
                       MasterDiv,MegaDivisionBoxes =  NMS.StaticEvents(MegaDivisionBoxes, DownsampleFactor)
                       NMS.drawimage(MasterApop, ResultsDirectory, fname, 'Apoptosis', 'Dynamic')
                       NMS.drawimage(MasterDiv, ResultsDirectory, fname, 'Division', 'Dynamic')
                       MasterApop = []
                       MasterDiv = []
       


def ClassicSmartONEAT(Moviefile, ResultsDirectory, ModelDirectory, ONEATA, ONEATB,  DownsampleFactor, multievent = True, TimeFrames = 7, Mode = 'Detection',
                      categories = 6, TrainshapeX = 54, TrainshapeY = 54, cut = 0.8, sizeTplus = 3, sizeTminus = 3, n_tiles = 2, densityveto = 10): 
    
    
      ConvertModel(ModelDirectory, ONEATA)
      ConvertModel(ModelDirectory, ONEATB)

      if classicNEAT:
                    
         NEATA =  load_model( ModelDirectory + ONEATA + '.h5',  custom_objects={'loss':mid_yolo_loss(categories), 'Concat':Concat})
         NEATB =  load_model( ModelDirectory + ONEATB + '.h5',  custom_objects={'loss':mid_yolo_loss(categories), 'Concat':Concat})
         
         
      if centerNEAT:
          
         NEATA =  load_model( ModelDirectory + ONEATA + '.h5',  custom_objects={'loss':cat_simple_yolo_loss(categories), 'Concat':Concat})
         NEATB =  load_model( ModelDirectory + ONEATB + '.h5',  custom_objects={'loss':cat_simple_yolo_loss(categories), 'Concat':Concat})
         
      if simplecenterNEAT:
          
         NEATA =  load_model( ModelDirectory + ONEATA + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
         NEATB =  load_model( ModelDirectory + ONEATB + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
         
      else:
         NEATA =  load_model( ModelDirectory + ONEATA + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
         NEATB =  load_model( ModelDirectory + ONEATB + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
          

      Categories_Name = []
      
      Categories_Name.append(['Normal', 0])
      Categories_Name.append(['Apoptosis', 1])
      Categories_Name.append(['Divisions', 2])
      Categories_Name.append(['MacroKitty', 3])
      Categories_Name.append(['NonMatureP1', 4])
      Categories_Name.append(['MatureP1', 5])
      
      fname = os.path.basename(os.path.splitext(Moviefile)[0])
      image = imread(Moviefile)
      
      scale_percent = int(100/DownsampleFactor) # percent of original size
      width = int(image.shape[2] * scale_percent / 100)
      height = int(image.shape[1] * scale_percent / 100)
      dim = (width, height)
      # resize image
      print('Sampling the image')
      smallimage = np.zeros([image.shape[0], height, width])
      for i in tqdm(range(0, image.shape[0])):
         smallimage[i,:] = cv2.resize(image[i,:], dim)
    
      image = smallimage 
      
      
      
      MegaDivisionBoxes = []
      MegaApoptosisBoxes = []
      
      MasterApop = []
      MasterDiv = []
      
      if classicNEAT:
          
         print('Running conventional ONEAT')
         
      else:
          
         print('Running center ONEAT') 
      for tlocation in tqdm(range(0, image.shape[0]- 1)):
           
           smallimg = CreateVolume(image, TimeFrames, tlocation,TrainshapeX, TrainshapeY,  Mode)  
         
           if smallimg.shape[0]==TimeFrames:
               
               PredictionEvents =  ONETLivePrediction(smallimg, NEATA, NEATB, DownsampleFactor, tlocation,
                                                    Categories_Name, TrainshapeX, TrainshapeY, TimeFrames, Mode, classic = classicNEAT, cut = cut, n_tiles = n_tiles, overlap_percent = 0.8 )
           
               PredictionEvents.GetLocationMaps()
               n_tiles = PredictionEvents.GetTiles()  
               LocationBoxesDivision = PredictionEvents.LocationBoxesDivision
               LocationBoxesApoptosis = PredictionEvents.LocationBoxesApoptosis
               
               
               if LocationBoxesApoptosis is not None: 
                   for j in range(0, len(LocationBoxesApoptosis)):
            
                     boxes, Label = LocationBoxesApoptosis[j]
                     MegaApoptosisBoxes.append(boxes)
                    
               if LocationBoxesDivision is not None:
                   for j in range(0, len(LocationBoxesDivision)):

                     boxes, Label = LocationBoxesDivision[j]
                     MegaDivisionBoxes.append(boxes)        
        
               if tlocation%TimeFrames == 0:
                       MasterApop, MegaApoptosisBoxes =  NMS.StaticEvents(MegaApoptosisBoxes, DownsampleFactor)
                       MasterDiv,MegaDivisionBoxes =  NMS.StaticEvents(MegaDivisionBoxes, DownsampleFactor)
                       NMS.drawimage(MasterApop, ResultsDirectory, fname, 'Apoptosis', 'Dynamic')
                       NMS.drawimage(MasterDiv, ResultsDirectory, fname, 'Division', 'Dynamic')
                       MasterApop = []
                       MasterDiv = []
                                   
                                   
                                   
def LiveSmartONEAT(image, fname, ResultsDirectory, NEATA, NEATB,  DownsampleFactor, multievent = False, 
                   TimeFrames = 4, Mode = 'Prediction',categories = 6, TrainshapeX = 54, TrainshapeY = 54, cut = 0.8, sizeTminus = 3, n_tiles = 2, densityveto = 10): 
    
        
        
    
           Categories_Name = []
              
           Categories_Name.append(['Normal', 0])
           Categories_Name.append(['Apoptosis', 1])
           Categories_Name.append(['Divisions', 2])
           Categories_Name.append(['MacroKitty', 3])
           Categories_Name.append(['NonMatureP1', 4])
           Categories_Name.append(['MatureP1', 5])
              
           print(image.shape)
              
           scale_percent = int(100/DownsampleFactor) # percent of original size
           width = int(image.shape[2] * scale_percent / 100)
           height = int(image.shape[1] * scale_percent / 100)
           dim = (width, height)
           # resize image
           print('Resizing the image')
           smallimage = np.zeros([image.shape[0], height, width])
           for i in tqdm(range(0, image.shape[0])): 
                 smallimage[i,:] = cv2.resize(image[i,:], dim)
            
           image = smallimage 
              
           MegaDivisionBoxes = []
           MegaApoptosisBoxes = []
            
           MasterApop = []
           MasterDiv = []

             
         
           
           smallimg = CreateVolume(image, TimeFrames, 0,TrainshapeX, TrainshapeY,  Mode)   
          
        
           if smallimg.shape[0]==TimeFrames:
               PredictionEvents =  ONETLivePrediction(smallimg, NEATA, NEATB, DownsampleFactor, 0,
                                                    Categories_Name, TrainshapeX, TrainshapeY, TimeFrames, Mode, classic = classicNEAT, cut = cut, n_tiles = n_tiles, overlap_percent = 0.8 )
           
               PredictionEvents.GetLocationMaps()
               n_tiles = PredictionEvents.GetTiles()  
               LocationBoxesDivision = PredictionEvents.LocationBoxesDivision
               LocationBoxesApoptosis = PredictionEvents.LocationBoxesApoptosis
               
               
               if LocationBoxesApoptosis is not None: 
                   for j in range(0, len(LocationBoxesApoptosis)):
            
                     boxes, Label = LocationBoxesApoptosis[j]
                     MegaApoptosisBoxes.append(boxes)
                    
               if LocationBoxesDivision is not None:
                   for j in range(0, len(LocationBoxesDivision)):

                     boxes, Label = LocationBoxesDivision[j]
                     
                     MegaDivisionBoxes.append(boxes)       
        
               
               
               MasterApop, MegaApoptosisBoxes =  NMS.NMSSpace(MegaApoptosisBoxes, DownsampleFactor)
               MasterDiv,MegaDivisionBoxes =  NMS.NMSSpace(MegaDivisionBoxes, DownsampleFactor)
               NMS.drawimage(MasterApop, ResultsDirectory, fname, 'Apoptosis', 'Dynamic')
               NMS.drawimage(MasterDiv, ResultsDirectory, fname, 'Division', 'Dynamic')
               MasterApop = []
               MasterDiv = []
                                   
                                   
def SmartPredONEAT(MovieDir, ResultCSVDirectory, NEATA, NEATB,  DownsampleFactor,MovieNameList, MovieInput, start,  multievent = False, 
                   TimeFrames = 4, Mode = 'Prediction',categories = 6, TrainshapeX = 54, TrainshapeY = 54, cut = 0.8, sizeTminus = 3, n_tiles = 2): 
    
    
      print('Models loaded, now listening to events')
      Categories_Name = []
      print('Starting Movie Number', start)
      print('Ending Movie Number', start + TimeFrames )
      Categories_Name.append(['Normal', 0])
      Categories_Name.append(['Apoptosis', 1])
      Categories_Name.append(['Divisions', 2])
      Categories_Name.append(['MacroKitty', 3])
      Categories_Name.append(['NonMatureP1', 4])
      Categories_Name.append(['MatureP1', 5])
      fname =  'Live'
      
      while 1:
             
              
              Raw_path = os.path.join(MovieDir, '*TIF')
              filesRaw = glob.glob(Raw_path)
              filesRaw = natsorted(filesRaw)
                
              for MovieName in filesRaw:  
                          Name = os.path.basename(os.path.splitext(MovieName)[0])
                          Extension = os.path.basename(os.path.splitext(MovieName)[1])
                          
                          #Check for unique filename
                          if Name not in MovieNameList and Extension == '.TIF':
                                  
                                  try:  
                                          
                                          
                                          image = imread(MovieName)
                                          MovieNameList.append(Name)
                                          shapeY = image.shape[0]
                                          shapeX = image.shape[1]

                                          MovieInput.append(image)

                                          TotalMovies = len(MovieInput)
                                          if TotalMovies >= TimeFrames + start:
                                              CurrentMovies = MovieInput[start:start + TimeFrames]
                                              print('Predicting on Movies:',MovieNameList[start:start + TimeFrames]) 
                                              PredictionImage = np.zeros([TimeFrames, shapeY, shapeX])
                                              for i in range(0, TimeFrames):
                                                  PredictionImage[i,:] = CurrentMovies[i]

                                              LiveSmartONEAT(PredictionImage, fname, ResultCSVDirectory, NEATA, NEATB,  DownsampleFactor, classicNEAT = classicNEAT,
                                               TimeFrames = TimeFrames, Mode = Mode,categories = categories, TrainshapeX = TrainshapeX, TrainshapeY = TrainshapeY, cut = cut, sizeTminus = sizeTminus, n_tiles = n_tiles)
                                              SmartPredONEAT(MovieDir, ResultCSVDirectory, NEATA, NEATB, DownsampleFactor, watcher,MovieNameList, MovieInput, start + 1, classicNEAT = classicNEAT, 
                                               TimeFrames = TimeFrames, Mode = Mode ,categories = categories, TrainshapeX = TrainshapeX, TrainshapeY = TrainshapeY, cut = cut, sizeTminus = sizeTminus, n_tiles = n_tiles)

                                  except:
                                     if Name in MovieNameList:
                                         MovieNameList.remove(Name) 
                                     pass
                     





def MatDynamicEvents(image,time, y, x, CubicModels, classicNEAT, Categories_Name, Category
                           ,Mode,n_tiles,TrainshapeX, TrainshapeY, TimeFrames ):
    

            
               MasterApop = []
               MasterDiv = []
                
               CubicModelA = CubicModels[0]
               CubicModelB = CubicModels[1]
    
    
               PredictionEvents =  ONETDynamicPrediction(image, CubicModelA, CubicModelB , 0,
                                                    Categories_Name, TrainshapeX, TrainshapeY, TimeFrames, Mode, cut = 0, classicNEAT = classicNEAT, n_tiles = n_tiles, overlap_percent = 0.8 )

           
               PredictionEvents.GetLocationMaps()
               n_tiles = PredictionEvents.GetTiles()  
               LocationBoxesApoptosis = PredictionEvents.LocationBoxesApoptosis
               LocationBoxesDivision = PredictionEvents.LocationBoxesDivision
               
               
               if len(LocationBoxesApoptosis) > 0:
                 boxA, _ = LocationBoxesApoptosis[0]
                
                 center = ( ((boxA[2])) , ((boxA[3])) )
       
                 size =  math.sqrt(boxA[8] * boxA[8] + boxA[9] * boxA[9] )
                 MasterApop.append([center, boxA[5],boxA[6], boxA[4], size] )
                 
               if len(LocationBoxesDivision) > 0:  
                  boxD, _ =  LocationBoxesDivision[0]
                  
                  center = ( ((boxD[2])) , ((boxD[3])) )
       
                  size =  math.sqrt(boxD[8] * boxD[8] + boxD[9] * boxD[9] )
                  MasterDiv.append([center, boxD[5],boxD[6], boxD[4], size] )
              
               Score = 0
               size = 0
               if Category == 1:
                   
                   #Check for apoptosis
                   if len(MasterApop) > 0:
                     location, time, Name, Score, size = MasterApop[0]
                   
               if Category == 2:
                   
                   #Check for division
                   if len(MasterDiv) > 0:
                      location, time, Name, Score, size = MasterDiv[0]
                
               
               return Score, size                                               
                
 
def SimpleSmartONEAT(Moviefile, Markerfile, ResultCSVDirectory, ModelDirectory, ModelA, ModelB, DownsampleFactor,
                     classicNEAT = False, centerNEAT = False, simplecenterNEAT = True, 
                     TimeFrames = 7, categories = 6, Mode = 'Detection', TrainshapeX = 54, TrainshapeY = 54, sizeTminus = 3, sizeTplus = 3, n_tiles = 1, densityveto = 10, batchsize = 100): 
    
    
    
      
      ConvertModel(ModelDirectory, ModelA)
      ConvertModel(ModelDirectory, ModelB)
      

      if classicNEAT:
          
          NEATModelA =  load_model( ModelDirectory + ModelA + '.h5',  custom_objects={'loss':mid_yolo_loss(categories), 'Concat':Concat})
          NEATModelB =  load_model( ModelDirectory + ModelB + '.h5',  custom_objects={'loss':mid_yolo_loss(categories), 'Concat':Concat})
 
      if centerNEAT:
          
          NEATModelA =  load_model( ModelDirectory + ModelA + '.h5',  custom_objects={'loss':cat_simple_yolo_loss(categories), 'Concat':Concat})
          NEATModelB =  load_model( ModelDirectory + ModelB + '.h5',  custom_objects={'loss':cat_simple_yolo_loss(categories), 'Concat':Concat})
      
      if simplecenterNEAT:
             
          NEATModelA =  load_model( ModelDirectory + ModelA + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
          NEATModelB =  load_model( ModelDirectory + ModelB + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
          
      else:
          
          
          NEATModelA =  load_model( ModelDirectory + ModelA + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
          NEATModelB =  load_model( ModelDirectory + ModelB + '.h5',  custom_objects={'loss':simple_yolo_loss(categories), 'Concat':Concat})
          
        
        
      Categories_Name = []
      CubicModels = [NEATModelA, NEATModelB]
      Categories_Name.append(['Normal', 0])
      Categories_Name.append(['Apoptosis', 1])
      Categories_Name.append(['Divisions', 2])
      Categories_Name.append(['MacroKitty', 3])
      Categories_Name.append(['NonMatureP1', 4])
      Categories_Name.append(['MatureP1', 5])
      
      fname = os.path.basename(os.path.splitext(Moviefile)[0])
      image = imread(Moviefile)
      MarkerImage = imread(Markerfile)
      
      scale_percent = int(100/DownsampleFactor) # percent of original size
      width = int(image.shape[2] * scale_percent / 100)
      height = int(image.shape[1] * scale_percent / 100)
      dim = (width, height)
      # resize image
      print('Resizing the image')
      smallimage = np.zeros([image.shape[0], height, width])
      for i in tqdm(range(0, image.shape[0])):
          smallimage[i,:] = cv2.resize(image[i,:], dim)
        
      image = smallimage 
      print('Making dictionary of markers')  
      AllTrees = MakeTrees(MarkerImage)
      print('Choosing Auto sampling factor for each region')  
      AllDensity = DensityCounter(MarkerImage, TrainshapeX, TrainshapeY)
      
      SpaceTime = MarkerToCSV(MarkerImage, TrainshapeX, TrainshapeY) 
      
     
     #os.remove(CSVfile)
      print('Making Predictions')
      
      for t in tqdm(range(0, len(SpaceTime))):
         time, cord = SpaceTime[t]
         
         for y, x in cord:
          
           if time > sizeTplus + 1 and math.isnan(x)==False and math.isnan(y)== False  :
              
           
               
                   x = x/DownsampleFactor
                   y = y/DownsampleFactor
                   crop_Xminus = x - int(TrainshapeX/2)
                   crop_Xplus = x  + int(TrainshapeX/2)
                   crop_Yminus = y  - int(TrainshapeY/2)
                   crop_Yplus = y  + int(TrainshapeY/2)
              
                   region =(slice(int(time - sizeTminus - 1),int(time + sizeTplus )),slice(int(crop_Yminus), int(crop_Yplus)),
                              slice(int(crop_Xminus), int(crop_Xplus)))
                   crop_image = image[region]
                   if(crop_image.shape[0] >= sizeTminus + sizeTplus + 1  and crop_image.shape[1] >= TrainshapeY - 1 and crop_image.shape[2] >= TrainshapeX - 1 ):
                           #out comes xy so reverse them to make yx instead
                           Apoptosis, Division  = SmartONEATEvents(crop_image,time, y, x, CubicModels, Categories_Name
                                           ,Mode,n_tiles,TrainshapeX, TrainshapeY, TimeFrames, classicNEAT,DownsampleFactor, densityveto = densityveto, batchsize = batchsize )
                           if len(Apoptosis) > 0:
                                   location, timeEvent, Name, Score, size = Apoptosis[0]
                                   location = (location[1] + crop_Yminus, location[0] + crop_Xminus)
        
                                   timelocation = time + timeEvent- sizeTminus - 1
                                  
                                   location = (location[0]*DownsampleFactor, location[1]*DownsampleFactor)
                                   
                                   tree, indices = AllTrees[str(int(timelocation))]
                                   distance, location = tree.query(location)
                                   location = int(indices[location][0]), int(indices[location][1]) 
        
                                   eventlist = []
                                   eventlist.append([(location[1] , location[0]), timelocation, Name, Score, size])
                                   drawimage(eventlist, ResultCSVDirectory, fname, Name, 'Dynamic')
                                   
                                   
                           if len(Division) > 0:
                                   location, timeEvent, Name, Score, size = Division[0]
                                   location = (location[1] + crop_Yminus, location[0] + crop_Xminus)
        
                                   timelocation = time + timeEvent- sizeTminus - 1
                                  
                                   location = (location[0]*DownsampleFactor, location[1]*DownsampleFactor)
                          
                                   tree, indices = AllTrees[str(int(timelocation))]
                                   distance, location = tree.query(location)
                                   location = int(indices[location][0]), int(indices[location][1]) 
        
                                   eventlist = []
                                   eventlist.append([(location[1] , location[0]), timelocation, Name, Score, size])
                                   drawimage(eventlist, ResultCSVDirectory, fname, Name, 'Dynamic')
                          
                                   
                                    
      for time in range(0, image.shape[0]):                              
                  density, originalbinlocation = AllDensity[str(int(time))]
                  for i in range(0, len(originalbinlocation)):
                           cord = originalbinlocation[i]
                           y, x = cord
                       
                           crop_Xminus = x - int(TrainshapeX/2)
                           crop_Xplus = x  + int(TrainshapeX/2)
                           crop_Yminus = y  - int(TrainshapeY/2)
                           crop_Yplus = y  + int(TrainshapeY/2)
                      
                           region =(slice(int(time - sizeTminus - 1),int(time + sizeTplus )),slice(int(crop_Yminus), int(crop_Yplus)),
                                      slice(int(crop_Xminus), int(crop_Xplus)))
                           crop_image = image[region]
                           if(crop_image.shape[0] >= sizeTminus + sizeTplus + 1  and crop_image.shape[1] >= TrainshapeY - 1 and crop_image.shape[2] >= TrainshapeX - 1 ):
                                   #out comes xy so reverse them to make yx instead
                                   Apoptosis, Division  = SmartONEATEvents(crop_image,time, y, x, CubicModels, Categories_Name
                                                   ,Mode,n_tiles,TrainshapeX, TrainshapeY, TimeFrames, classicNEAT, densityveto = densityveto, batchsize = batchsize )
                                   if len(Apoptosis) > 0:
                                           location, timeEvent, Name, Score, size = Apoptosis[0]
                                           location = (location[1] + crop_Yminus, location[0] + crop_Xminus)
                
                                           timelocation = time + timeEvent- sizeTminus - 1
                                          
                                           location = (location[0], location[1])
                                          
                                           tree, indices = AllTrees[str(int(timelocation))]
                                           distance, location = tree.query(location)
                                           location = int(indices[location][0]), int(indices[location][1]) 
                
                                           eventlist = []
                                           eventlist.append([(location[1] , location[0]), timelocation, Name, Score, size])
                                           drawimage(eventlist, ResultCSVDirectory, fname, Name, 'Dynamic')
                                         
                                   if len(Division) > 0:
                                           location, timeEvent, Name, Score, size = Division[0]
                                           location = (location[1] + crop_Yminus, location[0] + crop_Xminus)
                
                                           timelocation = time + timeEvent- sizeTminus - 1
                                          
                                           location = (location[0], location[1])
                                          
                                           tree, indices = AllTrees[str(int(timelocation))]
                                           distance, location = tree.query(location)
                                           location = int(indices[location][0]), int(indices[location][1]) 
            
                                           eventlist = []
                                           eventlist.append([(location[1] , location[0]), timelocation, Name, Score, size])
                                           drawimage(eventlist, ResultCSVDirectory, fname, Name, 'Dynamic')
           
               
    
def SmartONEATEvents(image,time, y, x, CubicModels, Categories_Name
                           ,Mode,n_tiles,TrainshapeX, TrainshapeY, TimeFrames, classicNEAT,DownsampleFactor, densityveto = 10, batchsize = 100 ):
    
    
            
               MasterApop = []
               MasterDiv = []
                
               CubicModelA = CubicModels[0]
               CubicModelB = CubicModels[1]
    
    
               PredictionEvents =  ONETDynamicPrediction(image, CubicModelA, CubicModelB , 0,
                                                    Categories_Name, TrainshapeX, TrainshapeY, TimeFrames, Mode, cut = 0.6, classicNEAT = classicNEAT, n_tiles = n_tiles, overlap_percent = 0.8, densityveto = densityveto, batchsize = batchsize )

           
               PredictionEvents.GetLocationMaps()
               n_tiles = PredictionEvents.GetTiles()  
               LocationBoxesApoptosis = PredictionEvents.LocationBoxesApoptosis
               LocationBoxesDivision = PredictionEvents.LocationBoxesDivision
            
               if time%TimeFrames == 0:
                       MasterApop, LocationBoxesApoptosis =  NMS.StaticEvents(LocationBoxesApoptosis, DownsampleFactor)
                       MasterDiv,LocationBoxesDivision =  NMS.StaticEvents(LocationBoxesDivision, DownsampleFactor)
               
                       if len(LocationBoxesApoptosis) > 0:
                         boxA, label = LocationBoxesApoptosis[0]

                         center = ( ((boxA[2])) , ((boxA[3])) )

                         size =  math.sqrt(boxA[8] * boxA[8] + boxA[9] * boxA[9] )
                         MasterApop.append([center, boxA[5],boxA[6], boxA[4], size] )

                       if len(LocationBoxesDivision) > 0:  
                          boxD, label =  LocationBoxesDivision[0]

                          center = ( ((boxD[2])) , ((boxD[3])) )

                          size =  math.sqrt(boxD[8] * boxD[8] + boxD[9] * boxD[9] )
                          MasterDiv.append([center, boxD[5],boxD[6], boxD[4], size] )
              
               
               return  MasterApop, MasterDiv                          
 

    
    

         
                                   

def StaticEvents(image, fname, CubicModels, Categories_Name, CutoffDistance
                           ,n_tiles, ResultsDirectory,TrainshapeX, TrainshapeY ):
    
    
       MegaMacroKittyBoxes = []
       MegaNonMatureBoxes = []
       MegaMatureBoxes = []
    
       MasterMacro = []
       MasterNonMature = []
       MasterMature = []
        
        
        
       CubicModelA = CubicModels[0]
       CubicModelB = CubicModels[1]
        
       MacroKittyCellSize = CutoffDistance[2]
       NonMatureCellSize = CutoffDistance[3]
       MatureCellSize = CutoffDistance[4]
       for i in tqdm(range(0, image.shape[0])):
            smallimage = image[i,:]
            PredictionEvents =  ONETStaticPrediction(smallimage, CubicModelA, CubicModelB ,
                                            Categories_Name, i,  TrainshapeX, TrainshapeY, "Detection", n_tiles = n_tiles, overlap_percent = 0.8 )

            PredictionEvents.GetLocationMaps()
            n_tiles = PredictionEvents.GetTiles()  
            LocationBoxesMacro = PredictionEvents.LocationBoxesMacro
            LocationBoxesNonMature = PredictionEvents.LocationBoxesNonMature
            LocationBoxesMature = PredictionEvents.LocationBoxesMature


            if LocationBoxesMacro is not None:
              for j in range(0, len(LocationBoxesMacro)):

                boxes, Label = LocationBoxesMacro[j]
                MegaMacroKittyBoxes.append(boxes)

            if LocationBoxesNonMature is not None:
              for j in range(0, len(LocationBoxesNonMature)):

                boxes, Label = LocationBoxesNonMature[j]
                MegaNonMatureBoxes.append(boxes)

            if LocationBoxesMature is not None:
              for j in range(0, len(LocationBoxesMature)):

                boxes, Label = LocationBoxesMature[j]
                MegaMatureBoxes.append(boxes)



            


            Trainshape = (TrainshapeX,TrainshapeY)
            MegaMacroKittyBoxes, MasterMacro = NMS.StaticEvents(MegaMacroKittyBoxes, Trainshape,MacroKittyCellSize, 'Static')
            NMS.drawimage(MasterMacro, ResultsDirectory, fname, i,  'Macrocheate', 'Static' )

            MegaNonMatureBoxes, MasterNonMature = NMS.StaticEvents(MegaNonMatureBoxes, Trainshape,NonMatureCellSize, 'Static' )
            NMS.drawimage(MasterNonMature,  ResultsDirectory, fname, i, 'NonMature', 'Static' )

            MegaMatureBoxes, MasterMature = NMS.StaticEvents(MegaMatureBoxes, Trainshape,MatureCellSize, 'Static' )
            NMS.drawimage(MasterMature, ResultsDirectory, fname, i, 'Mature', 'Static' )





def CreateVolume(image, TimeFrames, TimePoint, TrainshapeY, TrainshapeX, Mode):
    finalend = image.shape[0]
    if Mode == "Detection" or Mode == "ShortDetection":    
        
               
            
               starttime = TimePoint
               endtime = TimePoint + TimeFrames
               if endtime > finalend:
                   endtime = finalend
                   starttime = finalend - TimeFrames
               smallimg = image[starttime:endtime, :]
        
    else:
                   
            smallimg = image[0:TimeFrames, :]
    
       
    return smallimg         
    

    
    
        

    
    
        
