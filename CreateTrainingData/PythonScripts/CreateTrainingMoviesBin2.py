#!/usr/bin/env python
# coding: utf-8

# # Create training movies from csv file
# In this notebook, we create training movies using the csv files generated by our MouseClick Fiji plugin which writes the event location as t,x,y,boolean format. The user has to input the image path used for creating the csv file along with the csv file. Other parameters to be specified by the user are indicated preceeding their respective blocks.

# In[ ]:


import numpy as np
from tifffile import imread 
import sys
import os
from glob import glob
sys.path.append("../NEAT")
from  NEATUtils import Augmentation
from NEATUtils import MovieCreator, npzfileGenerator
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


# In the block below specify the image used fore creating the csv files along with the total categories for classification. 
# 
# The default categories are: Norman events , Apoptosis, Division, Macrocheate, NonMature and Mature P1 cells. 
# 
# Crop size specifies the XY crop to be created at the location of the dot in the csv file and the third dimension is the crop in time relative to the click location, so it is X,Y,T, Crops image area sizeX = -X/2:X/2, -Y/2:Y/2, -T:T, so 7 Frame movies if T = 3
# 
# Since the network learns the joint task of classification and localization, we need to create that anwser int eh training data itself. Hence user specifies the value of shift in pixels to shift the event to, the program moves the event location from the center to nine locations (left, right, bottom, top, upper-left/right and bottom-left/right)
# 
# The task of the program below is to create movie crops and assigned training label to each crop, labels used for events are:
# 
# Normal Event: Label 0
# Apoptosis Event: Label 1
# Division Event: Label 2
# Macrocheate Event: Label 3
# Non MatureP1 Event : Label 4
# Mature Event: Label 5
# 
# So for division in the center the training label would be
# [0,0,1,0,0,0.5,0.5,0.5]
# for the same event on the top Left
# [0,0,1,0,0,0.38,0.38,0.5] and so on. This task is done by the program automatically

# In[ ]:


###### Specify the image used for making the csv file
SourceDir = '/home/sancere/VarunNewton/TrainingDataONEAT/CSVforNeat/NEATcsvfiles/'
SourceImage = imread(SourceDir + 'wt_N10.tif')
NormalCSV = SourceDir + 'Movie2Normal.csv'
ApoptosisCSV = SourceDir +  'wt_N10Apoptosis.csv'
DivisionCSV = SourceDir +  'wt_N10Division.csv'
MacrocheateCSV = SourceDir +  'Movie2MacroKitty.csv'
NonMatureCSV = SourceDir +  'Movie2NonMature.csv'
MatureCSV = SourceDir +  'Movie2Mature.csv'

Rawsave_dir = '/home/sancere/VarunNewton/TrainingDataONEAT/'
RawNormalsave_dir = Rawsave_dir + 'MidDynamicNormalEventTrainingData'
RawApoptosissave_dir = Rawsave_dir + 'MidDynamicMasterApoptosisEventTrainingData'
RawDivisionsave_dir = Rawsave_dir + 'MidDynamicMasterDivisionEventTrainingData'
RawMacrocheatesave_dir = Rawsave_dir + 'MidDynamicMacroKittyEventTrainingData'
RawNonMaturesave_dir = Rawsave_dir + 'MidDynamicNonMatureP1EventTrainingData'
RawMaturesave_dir = Rawsave_dir + 'MidDynamicMatureP1EventTrainingData'

Path(RawNormalsave_dir).mkdir(exist_ok = True)
Path(RawApoptosissave_dir).mkdir(exist_ok = True)
Path(RawDivisionsave_dir).mkdir(exist_ok = True)
Path(RawMacrocheatesave_dir).mkdir(exist_ok = True)
Path(RawNonMaturesave_dir).mkdir(exist_ok = True)
Path(RawMaturesave_dir).mkdir(exist_ok = True)

Localizationsave_dir = '/home/sancere/VarunNewton/CurieTrainingDatasets/Raw_Datasets/Neat/'
LocalizationNormalsave_dir = Localizationsave_dir +  'MidDynamicNormalEventTrainingData'
LocalizationApoptosissave_dir = Localizationsave_dir + 'MidDynamicMasterApoptosisEventTrainingData'
LocalizationDivisionsave_dir = Localizationsave_dir + 'MidDynamicMasterDivisionEventTrainingData'
LocalizationMacrocheatesave_dir = Localizationsave_dir + 'MidDynamicMacroKittyEventTrainingData'
LocalizationNonMaturesave_dir = Localizationsave_dir + 'MidDynamicNonMatureP1EventTrainingData'
LocalizationMaturesave_dir = Localizationsave_dir +  'MidDynamicMatureP1EventTrainingData'

Path(LocalizationNormalsave_dir).mkdir(exist_ok = True)
Path(LocalizationApoptosissave_dir).mkdir(exist_ok = True)
Path(LocalizationDivisionsave_dir).mkdir(exist_ok = True)
Path(LocalizationMacrocheatesave_dir).mkdir(exist_ok = True)
Path(LocalizationNonMaturesave_dir).mkdir(exist_ok = True)
Path(LocalizationMaturesave_dir).mkdir(exist_ok = True)

SaveNpzDirectory = '/home/sancere/VarunNewton/CurieTrainingDatasets/O-NEAT/'

crop_size = [64,64,3]
SizeX = crop_size[0]
SizeY = crop_size[1]
#Shift the event by these many pixels
shift = 10
TotalCategories = 6


# In[ ]:



#For Normal Events/Negative controls
MovieCreator.CreateMoviesTXYZ(NormalCSV, SourceImage, crop_size, 0,TotalCategories, 0 ,RawNormalsave_dir, 'ONEAT')

#For Macrocheate
MovieCreator.CreateMoviesTXYZ(MacrocheateCSV, SourceImage, crop_size, shift,TotalCategories, 3 ,RawMacrocheatesave_dir, 'ONEAT')
#For NonMatureP1
MovieCreator.CreateMoviesTXYZ(NonMatureCSV, SourceImage, crop_size, shift,TotalCategories, 4 ,RawNonMaturesave_dir, 'ONEAT')
#For MatureP1
MovieCreator.CreateMoviesTXYZ(MatureCSV, SourceImage, crop_size, shift,TotalCategories, 5 ,RawMaturesave_dir, 'ONEAT')
#For Apoptosis
MovieCreator.CreateMoviesTXYZ(ApoptosisCSV, SourceImage, crop_size, shift,TotalCategories, 1 ,RawApoptosissave_dir, 'ONEAT')
#For Division
MovieCreator.CreateMoviesTXYZ(DivisionCSV, SourceImage, crop_size, shift,TotalCategories, 2 ,RawDivisionsave_dir, 'ONEAT')


# In[ ]:


#elasticDeform = True, putNoise (Blur) = True in that order
Subdir = next(os.walk(RawNormalsave_dir))

for x in Subdir[1]:
    currentdir = RawNormalsave_dir + '/' + x
    Augmentation(currentdir,LocalizationNormalsave_dir +'/' + x, SizeX, SizeY, False,True,AppendName = 'Master')
    
Subdir = next(os.walk(RawApoptosissave_dir))

for x in Subdir[1]:
    
    currentdir = RawApoptosissave_dir + '/' + x
    Augmentation(currentdir,LocalizationApoptosissave_dir +'/' + x, SizeX, SizeY, False,True,AppendName = 'Master') 
    
Subdir = next(os.walk(RawDivisionsave_dir))

for x in Subdir[1]:
    
    currentdir = RawDivisionsave_dir + '/' + x
    Augmentation(currentdir,LocalizationDivisionsave_dir +'/' + x, SizeX, SizeY, False,False,AppendName = 'Master') 
    
Subdir = next(os.walk(RawMacrocheatesave_dir))

for x in Subdir[1]:
    
    currentdir = RawMacrocheatesave_dir + '/' + x
    Augmentation(currentdir,LocalizationMacrocheatesave_dir +'/' + x, SizeX, SizeY, False,True,AppendName = 'Master') 

Subdir = next(os.walk(RawNonMaturesave_dir))

for x in Subdir[1]:
    
    currentdir = RawNonMaturesave_dir + '/' + x
    Augmentation(currentdir,LocalizationNonMaturesave_dir +'/' + x, SizeX, SizeY, False,True,AppendName = 'Master') 
    
Subdir = next(os.walk(RawMaturesave_dir))

for x in Subdir[1]:
    
    currentdir = RawMaturesave_dir + '/' + x
    Augmentation(currentdir,LocalizationMaturesave_dir +'/' + x, SizeX, SizeY, False,True,AppendName = 'Master') 
        
    


# In[ ]:


DirectoryList = []
LabelList = []

Subdir = next(os.walk(LocalizationNormalsave_dir))

for x in Subdir[1]:
    currentdir = LocalizationNormalsave_dir + '/' + x
    CsvFile = sorted(glob(currentdir + '/' + '*.csv'))
    Labels = np.loadtxt(CsvFile[0], unpack = True)
    SubSubdir = next(os.walk(currentdir))
    for y in SubSubdir[1]:
        alldir = LocalizationNormalsave_dir + '/' + x + '/' + y+ '/'
        print(alldir)
        DirectoryList.append(alldir)
        LabelList.append(Labels)
        
Subdir = next(os.walk(LocalizationApoptosissave_dir))

for x in Subdir[1]:
    currentdir = LocalizationApoptosissave_dir + '/' + x 
    CsvFile = sorted(glob(currentdir + '/' + '*.csv'))
    
    Labels = np.loadtxt(CsvFile[0], unpack = True)
    SubSubdir = next(os.walk(currentdir))
    for y in SubSubdir[1]:
        alldir = LocalizationApoptosissave_dir + '/' + x + '/' + y+ '/'
        DirectoryList.append(alldir)
        LabelList.append(Labels)
        
Subdir = next(os.walk(LocalizationDivisionsave_dir))

for x in Subdir[1]:
    currentdir = LocalizationDivisionsave_dir + '/' + x 
    CsvFile = sorted(glob(currentdir + '/' + '*.csv'))
    Labels = np.loadtxt(CsvFile[0], unpack = True)
    SubSubdir = next(os.walk(currentdir))
    for y in SubSubdir[1]:
        alldir = LocalizationDivisionsave_dir + '/' + x + '/' + y+ '/'
        DirectoryList.append(alldir)
        LabelList.append(Labels)
        
Subdir = next(os.walk(LocalizationMacrocheatesave_dir))

for x in Subdir[1]:
    currentdir = LocalizationMacrocheatesave_dir + '/' + x 
    CsvFile = sorted(glob(currentdir + '/' + '*.csv'))
    Labels = np.loadtxt(CsvFile[0], unpack = True)
    SubSubdir = next(os.walk(currentdir))
    for y in SubSubdir[1]:
        alldir = LocalizationMacrocheatesave_dir + '/' + x + '/' + y+ '/'
        DirectoryList.append(alldir)
        LabelList.append(Labels)
        
Subdir = next(os.walk(LocalizationNonMaturesave_dir))

for x in Subdir[1]:
    currentdir = LocalizationNonMaturesave_dir + '/' + x 
    CsvFile = sorted(glob(currentdir + '/' + '*.csv'))
    Labels = np.loadtxt(CsvFile[0], unpack = True)
    SubSubdir = next(os.walk(currentdir))
    for y in SubSubdir[1]:
        alldir = LocalizationNonMaturesave_dir + '/' + x + '/' + y+ '/'
        DirectoryList.append(alldir)
        LabelList.append(Labels)
        
Subdir = next(os.walk(LocalizationMaturesave_dir))

for x in Subdir[1]:
    currentdir = LocalizationMaturesave_dir + '/' + x 
    CsvFile = sorted(glob(currentdir + '/' + '*.csv'))
    Labels = np.loadtxt(CsvFile[0], unpack = True)
    SubSubdir = next(os.walk(currentdir))
    for y in SubSubdir[1]:
        alldir = LocalizationMaturesave_dir + '/' + x + '/' + y+ '/'
        DirectoryList.append(alldir)
        LabelList.append(Labels) 
        


# In[ ]:


SaveName = 'MasterONEAT'
SaveNameVal = 'MasterONEATValidation'


MovieFrames = 7
npzfileGenerator.generate_training_data(DirectoryList, LabelList,SaveNpzDirectory, SaveName, SaveNameVal,0, MovieFrames, SizeX, SizeY)
        
SaveName = 'MasterONEATPrediction'
SaveNameVal = 'MasterONEATPredictionValidation'


MovieFrames = 4
npzfileGenerator.generate_training_data(DirectoryList, LabelList,SaveNpzDirectory, SaveName, SaveNameVal,0, MovieFrames, SizeX, SizeY)
            


# In[ ]:




