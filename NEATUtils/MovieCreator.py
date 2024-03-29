import csv
import numpy as np
from tifffile import imread, imwrite 
import pandas as pd
import os
import glob
from skimage.measure import regionprops
from skimage import measure
from scipy import spatial 
from pathlib import Path
from .helpers import  normalizeFloatZeroOne
    
"""
@author: Varun Kapoor
In this program we create training movies and training images for ONEAT. The training data comprises of images and text labels attached to them.
TrainingMovies: This program is for action recognition training data creation. The inputs are the training image, the corresponding integer labelled segmentation image,
csv file containing time, ylocation, xlocation, angle (optional)
Additional parameters to be supplied are the 
1) sizeTminus: action events are centered at the time location, this parameter is the start time of the time volume the network carved out from the image.
2) sizeTplus: this parameter is the end of the time volume to be carved out from the image.
3) TotalCategories: It is the number of total action categories the network is supposed to predict, Vanilla ONEAT has these labels:
   0: NormalEvent
   1: ApoptosisEvent
   2: DivisionEvent
   3: Macrocheate as static dynamic event
   4: Non MatureP1 cells as static dynamic event
   5: MatureP1 cells as static dynamic event
    
TrainingImages: This program is for cell type recognition training data creation. The inputs are the trainng image, the corresponding integer labelled segmentation image,
Total categories for cell classification part of vanilla ONEAT are:
    0: Normal cells
    1: Central time frame of apoptotic cell
    2: Central time frame of dividing cell
    3: Macrocheates
    4: Non MatureP1 cells
    5: MatureP1 cells
csv file containing time, ylocation, xlocation of that event/cell type
"""    
    
def MovieLabelDataSet(ImageDir, SegImageDir, CSVDir,SaveDir, StaticName, StaticLabel, CSVNameDiff,crop_size, gridX = 1, gridY = 1, offset = 0):
    
    
            Raw_path = os.path.join(ImageDir, '*tif')
            Seg_path = os.path.join(SegImageDir, '*tif')
            Csv_path = os.path.join(CSVDir, '*csv')
            filesRaw = glob.glob(Raw_path)
            filesRaw.sort
            filesSeg = glob.glob(Seg_path)
            filesSeg.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(SaveDir).mkdir(exist_ok=True)
            TotalCategories = len(StaticName)
            
            for csvfname in filesCsv:
              count = 0  
              print(csvfname)
              CsvName =  os.path.basename(os.path.splitext(csvfname)[0])
            
              for fname in filesRaw:
                  
                 Name = os.path.basename(os.path.splitext(fname)[0])   
                 for Segfname in filesSeg:
                      
                      SegName = os.path.basename(os.path.splitext(Segfname)[0])
                        
                      if Name == SegName:
                          
                          
                         image = imread(fname)
                         segimage = imread(Segfname)
                         for i in  range(0, len(StaticName)):
                             Eventname = StaticName[i]
                             trainlabel = StaticLabel[i]
                             if CsvName == CSVNameDiff + Name + Eventname:
                                            dataset = pd.read_csv(csvfname)
                                            if len(dataset.keys() >= 3):
                        
                                                time = dataset[dataset.keys()[0]][1:]
                                                y = dataset[dataset.keys()[1]][1:]
                                                x = dataset[dataset.keys()[2]][1:]
                                                angle = np.full(time.shape, 2)                        
                                            if len(dataset.keys() > 3):
                                                
                                                angle = dataset[dataset.keys()[3]][1:]                          
                                            #Categories + XYHW + Confidence 
                                            for t in range(1, len(time)):
                                               MovieMaker(time[t], y[t], x[t], angle[t], image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, Name + Eventname + str(count), SaveDir)
                                               count = count + 1
                                               


def CreateTrainingMovies(csv_file, image, segimage, crop_size, TotalCategories, trainlabel, save_dir, gridX = 1, gridY = 1, offset = 0, defname = "" ):

            Path(save_dir).mkdir(exist_ok=True)
            name = 1
            #Check if the csv file exists
            if os.path.exists(csv_file):
                    dataset = pd.read_csv(csv_file)
                    # The csv files contain TYX or TYX + Angle
                    if len(dataset.keys() >= 3):
                        
                        time = dataset[dataset.keys()[0]][1:]
                        y = dataset[dataset.keys()[1]][1:]
                        x = dataset[dataset.keys()[2]][1:]
                        angle = np.full(time.shape, 2)                        
                    if len(dataset.keys() > 3):
                        
                        angle = dataset[dataset.keys()[3]][1:]      
                    
                    #Categories + XYTHW + Confidence + Angle
                    for t in time:
                       MovieMaker(time[t], y[t], x[t], angle[t], image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, defname + str(name), save_dir)
                       name = name + 1
               

            
def MovieMaker(time, y, x, angle, image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, name, save_dir):
    
       sizeX, sizeY, sizeTminus, sizeTplus = crop_size
       
       ImagesizeX = sizeX * gridX
       ImagesizeY = sizeY * gridY
       
       shiftNone = [0,0]
       if offset > 0 and trainlabel > 0:
                 shiftLX = [int(offset), 0] 
                 shiftRX = [-offset, 0]
                 shiftLXY = [int(offset), int(offset)]
                 shiftRXY = [-int(offset), int(offset)]
                 shiftDLXY = [int(offset), -int(offset)]
                 shiftDRXY = [-int(offset), -int(offset)]
                 shiftUY = [0, int(offset)]
                 shiftDY = [0, -int(offset)]
                 AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY]

       else:
           
          AllShifts = [shiftNone]


       
       currentsegimage = segimage[time,:].astype('uint16')
       height, width, center, SegLabel = getHW(x, y, trainlabel, currentsegimage)
       for shift in AllShifts:
           
                newname = name + 'shift' + str(shift)
                Event_data = []
                newcenter = (center[0] - shift[1],center[1] - shift[0] )
                x = center[1]
                y = center[0]
                Label = np.zeros([TotalCategories + 7])
                Label[trainlabel] = 1
                #T co ordinate
                Label[TotalCategories + 2] = (sizeTminus) / (sizeTminus + sizeTplus)
                if x + shift[0]> sizeX/2 and y + shift[1] > sizeY/2 and x + shift[0] + int(ImagesizeX/2) < image.shape[2] and y + shift[1]+ int(ImagesizeY/2) < image.shape[1] and time > sizeTminus and time + sizeTplus + 1 < image.shape[0]:
                        crop_Xminus = x  - int(ImagesizeX/2)
                        crop_Xplus = x  + int(ImagesizeX/2)
                        crop_Yminus = y  - int(ImagesizeY/2)
                        crop_Yplus = y   + int(ImagesizeY/2)
                        # Cut off the region for training movie creation
                        region =(slice(int(time - sizeTminus),int(time + sizeTplus  + 1)),slice(int(crop_Yminus)+ shift[1], int(crop_Yplus)+ shift[1]),
                              slice(int(crop_Xminus) + shift[0], int(crop_Xplus) + shift[0]))
                        #Define the movie region volume that was cut
                        crop_image = image[region]   
                        crop_image =  normalizeFloatZeroOne(crop_image ,1,99.8)
                        seglocationX = (newcenter[1] - crop_Xminus)
                        seglocationY = (newcenter[0] - crop_Yminus)
                         
                        Label[TotalCategories] =  seglocationX/sizeX
                        Label[TotalCategories + 1] = seglocationY/sizeY
                        
                        #Height
                        Label[TotalCategories + 3] = height/ImagesizeY
                        #Width
                        Label[TotalCategories + 4] = width/ImagesizeX
               
                          
                        Label[TotalCategories + 5] = angle  
                        
                        if SegLabel > 0:
                          Label[TotalCategories + 6] = 1 
                        else:
                          Label[TotalCategories + 6] = 0   
                      
                        #Write the image as 32 bit tif file 
                        if(crop_image.shape[0] == sizeTplus + sizeTminus + 1 and crop_image.shape[1]== ImagesizeY and crop_image.shape[2]== ImagesizeX):
                                  
                                   imwrite((save_dir + '/' + newname + '.tif'  ) , crop_image.astype('float32'))    
                                   Event_data.append([Label[i] for i in range(0,len(Label))])
                                   if(os.path.exists(save_dir + '/' + (newname) + ".csv")):
                                                os.remove(save_dir + '/' + (newname) + ".csv")
                                   writer = csv.writer(open(save_dir + '/' + (newname) + ".csv", "a"))
                                   writer.writerows(Event_data)

       
   
                  
def ImageLabelDataSet(ImageDir, SegImageDir, CSVDir,SaveDir, StaticName, StaticLabel, CSVNameDiff,crop_size, gridX = 1, gridY = 1, offset = 0):
    
    
            Raw_path = os.path.join(ImageDir, '*tif')
            Seg_path = os.path.join(SegImageDir, '*tif')
            Csv_path = os.path.join(CSVDir, '*csv')
            filesRaw = glob.glob(Raw_path)
            filesRaw.sort
            filesSeg = glob.glob(Seg_path)
            filesSeg.sort
            filesCsv = glob.glob(Csv_path)
            filesCsv.sort
            Path(SaveDir).mkdir(exist_ok=True)
            TotalCategories = len(StaticName)
            
            for csvfname in filesCsv:
              print(csvfname)
              count = 0
              CsvName =  os.path.basename(os.path.splitext(csvfname)[0])
            
              for fname in filesRaw:
                  
                 Name = os.path.basename(os.path.splitext(fname)[0])   
                 for Segfname in filesSeg:
                      
                      SegName = os.path.basename(os.path.splitext(Segfname)[0])
                        
                      if Name == SegName:
                          
                          
                         image = imread(fname)
                         segimage = imread(Segfname)
                         for i in  range(0, len(StaticName)):
                             Eventname = StaticName[i]
                             trainlabel = StaticLabel[i]
                             if CsvName == CSVNameDiff + Name + Eventname:
                                            dataset = pd.read_csv(csvfname)
                                            time = dataset[dataset.keys()[0]][1:]
                                            y = dataset[dataset.keys()[1]][1:]
                                            x = dataset[dataset.keys()[2]][1:]     
                                            
                                            #Categories + XYHW + Confidence 
                                            for t in range(1, len(time)):
                                               ImageMaker(time[t], y[t], x[t], image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, Name + Eventname + str(count), SaveDir)    
                                               count = count + 1
    



def  ImageMaker(time, y, x, image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, name, save_dir):

               sizeX, sizeY = crop_size

               ImagesizeX = sizeX * gridX
               ImagesizeY = sizeY * gridY

               shiftNone = [0,0]
               if offset > 0 and trainlabel > 0:
                 shiftLX = [int(offset), 0] 
                 shiftRX = [-offset, 0]
                 shiftLXY = [int(offset), int(offset)]
                 shiftRXY = [-int(offset), int(offset)]
                 shiftDLXY = [int(offset), -int(offset)]
                 shiftDRXY = [-int(offset), -int(offset)]
                 shiftUY = [0, int(offset)]
                 shiftDY = [0, -int(offset)]
                 AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY]

               else:

                  AllShifts = [shiftNone]

       
               if time < segimage.shape[0] - 1:
                 currentsegimage = segimage[int(time),:].astype('uint16')
                
                 height, width, center, SegLabel  = getHW(x, y,trainlabel, currentsegimage)
                 for shift in AllShifts:
                   
                        newname = name + 'shift' + str(shift)
                        newcenter = (center[0] - shift[1],center[1] - shift[0] )
                        Event_data = []
                        
                        x = center[1]
                        y = center[0]
                        Label = np.zeros([TotalCategories + 5])
                        Label[trainlabel] = 1
                        if x + shift[0]> sizeX/2 and y + shift[1] > sizeY/2 and x + shift[0] + int(ImagesizeX/2) < image.shape[2] and y + shift[1]+ int(ImagesizeY/2) < image.shape[1]:
                                    crop_Xminus = x  - int(ImagesizeX/2)
                                    crop_Xplus = x   + int(ImagesizeX/2)
                                    crop_Yminus = y  - int(ImagesizeY/2)
                                    crop_Yplus = y   + int(ImagesizeY/2)
                                 
                                    
                                    region =(slice(int(time - 1),int(time)),slice(int(crop_Yminus)+ shift[1], int(crop_Yplus)+ shift[1]),
                                           slice(int(crop_Xminus) + shift[0], int(crop_Xplus) + shift[0]))
                                   
                                    crop_image = image[region]      
                                    crop_image =  normalizeFloatZeroOne(crop_image ,1,99.8)
                                    seglocationX = (newcenter[1] - crop_Xminus)
                                    seglocationY = (newcenter[0] - crop_Yminus)
                                      
                                    Label[TotalCategories] =  seglocationX/sizeX
                                    Label[TotalCategories + 1] = seglocationY/sizeY
                                    
                                    
                                    Label[TotalCategories + 2] = height/ImagesizeY
                                    Label[TotalCategories + 3] = width/ImagesizeX
                                   

                                    if SegLabel > 0:
                                      Label[TotalCategories + 4] = 1 
                                    else:
                                      Label[TotalCategories + 4] = 0  
                                 
                                    if(crop_image.shape[1]== ImagesizeY and crop_image.shape[2]== ImagesizeX):
                                             imwrite((save_dir + '/' + newname + '.tif'  ) , crop_image.astype('float32'))  
                                             Event_data.append([Label[i] for i in range(0,len(Label))])
                                             if(os.path.exists(save_dir + '/' + (newname) + ".csv")):
                                                os.remove(save_dir + '/' + (newname) + ".csv")
                                             writer = csv.writer(open(save_dir + '/' + (newname) + ".csv", "a"))
                                             writer.writerows(Event_data)

       
def getHW(defaultX, defaultY, trainlabel, currentsegimage):
    
    properties = measure.regionprops(currentsegimage, currentsegimage)
    TwoDLocation = (int(defaultY),int(defaultX))
    TwoDCoordinates = [(prop.centroid[0], prop.centroid[1]) for prop in properties]
    TwoDtree = spatial.cKDTree(TwoDCoordinates)
    closestpoint = TwoDtree.query(TwoDLocation)
    ClosestLocation = ( int(TwoDCoordinates[closestpoint[1]][0]), int(TwoDCoordinates[closestpoint[1]][1]))
    SegLabel = currentsegimage[TwoDLocation]
    if trainlabel > 0 and SegLabel == 0:
           SegLabel = currentsegimage[ClosestLocation]
    for prop in properties:
                                               
                  if SegLabel > 0 and prop.label == SegLabel:
                                    minr, minc, maxr, maxc = prop.bbox
                                    center = prop.centroid
                                    height =  abs(maxc - minc)
                                    width =  abs(maxr - minr)
                                
                  if SegLabel == 0:
                    
                             center = (defaultY, defaultX)
                             height = 10
                             width = 10
                    
                                
    return height, width, center, SegLabel      


def  AngleAppender(AngleCSV, ONTCSV, save_dir, ColumnA = 'Y'):

     dataset = pd.read_csv(AngleCSV)
     time = dataset[dataset.keys()[0]][1:]
     if ColumnA == 'Y':
       y = dataset[dataset.keys()[1]][1:]
       x = dataset[dataset.keys()[2]][1:]
       angle = dataset[dataset.keys()[2]][1:]
     else:
       x = dataset[dataset.keys()[1]][1:]
       y = dataset[dataset.keys()[2]][1:]
       angle = dataset[dataset.keys()[2]][1:]  
       
     clickeddataset = pd.read_csv(ONTCSV)  
     clickedtime = clickeddataset[clickeddataset.keys()[0]][1:]
     clickedy = clickeddataset[clickeddataset.keys()[1]][1:]
     clickedx = clickeddataset[clickeddataset.keys()[2]][1:]
                               
     Event_data = []
     
     Name = os.path.basename(os.path.splitext(ONTCSV)[0])
     for clickedt in range(1, len(clickedtime)):
                          
                for t in range(1, len(time)): 

                          if time[t] == clickedtime[clickedt] and y[t] == clickedy[clickedt] and x[t] == clickedx[clickedt]:
                              
                                Event_data.append([time[t],y[t],x[t],angle[t]])
                              
                                     
                              
                               
     writer = csv.writer(open(save_dir + '/' + (Name) + ".csv", "a"))
     writer.writerows(Event_data)
     
                      
