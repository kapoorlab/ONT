#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from tifffile import imread 
import sys
import os
import cv2
from glob import glob
from tqdm import tqdm
sys.path.append("../NEAT")
from NEATUtils import MovieCreator
try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path

try:
    import tempfile
    tempfile.TemporaryDirectory
except (ImportError,AttributeError):
    from backports import tempfile


# In[ ]:


###### Specify the image used for making the csv file
SourceDir = '/data/u934/service_imagerie/v_kapoor/CSVforNeat/YolONEAT/TrainCSV/'

CenterTrainDataDir = '/data/u934/service_imagerie/v_kapoor/CSVforNeat/YolONEAT/CenterTrainData/'
GenericTrainDataDir = '/data/u934/service_imagerie/v_kapoor/CSVforNeat/YolONEAT/GenericTrainData/'

StaticCenterTrainDataDir = '/data/u934/service_imagerie/v_kapoor/CSVforNeat/YolONEAT/StaticCenterTrainData/'
StaticGenericTrainDataDir = '/data/u934/service_imagerie/v_kapoor/CSVforNeat/YolONEAT/StaticGenericTrainData/'

MovieName = 'EventMovie'
SegMovieName = 'SegEventMovie'
SourceImage = imread(SourceDir + MovieName + '.tif')
SegmentationImage = imread(SourceDir + SegMovieName + '.tif')

NormalCSV = SourceDir + MovieName +'Normal.csv'
ApoptosisCSV = SourceDir + MovieName +'Apoptosis.csv'
DivisionCSV = SourceDir + MovieName +'Division.csv'
MacrocheateCSV = SourceDir + MovieName +'MacroKitty.csv'
NonMatureCSV = SourceDir + MovieName +'NonMature.csv'
MatureCSV = SourceDir + MovieName +'Mature.csv'


# In[ ]:


#X Y Tminus and Tplus for making image volumes
crop_size = [256,256,5,3]
static_crop_size = [crop_size[0], crop_size[1]]
SizeX = crop_size[0]
SizeY = crop_size[1]
gridX = 1
gridY = 1
TotalCategories = 6

#For creating trainng data with shifted events, centerONEAT does not use this offset
offset = 10


# In[ ]:


EventCSV = [NormalCSV, ApoptosisCSV, DivisionCSV,MacrocheateCSV,NonMatureCSV,MatureCSV]
EventTrain = [0, 1, 2, 3, 4, 5]
TotalCategories = len(EventCSV)
print('Categories for training', TotalCategories)


# # For center ONEAT, event is exactly in the center for all training examples

# In[ ]:


for i in range(TotalCategories):
     csv_file = EventCSV[i]
     trainlabel = EventTrain[i]   
     MovieCreator.CreateTrainingMovies(csv_file, SourceImage, SegmentationImage, crop_size, TotalCategories, trainlabel, CenterTrainDataDir, gridX = gridX, gridY = gridY)
     MovieCreator.CreateTrainingImages(csv_file, SourceImage, SegmentationImage, static_crop_size, TotalCategories, trainlabel, StaticCenterTrainDataDir, gridX = gridX, gridY = gridY)   


# # For non-center ONEAT we shift the training event by offset pixels

# In[ ]:



for i in range(TotalCategories):
     csv_file = EventCSV[i]
     trainlabel = EventTrain[i]   
     MovieCreator.CreateTrainingMovies(csv_file, SourceImage, SegmentationImage, crop_size, TotalCategories, trainlabel, GenericTrainDataDir, gridX = gridX, gridY = gridY, offset = offset)
     MovieCreator.CreateTrainingImages(csv_file, SourceImage, SegmentationImage, static_crop_size, TotalCategories, trainlabel, StaticGenericTrainDataDir, gridX = gridX, gridY = gridY, offset = offset)      


# In[ ]:




