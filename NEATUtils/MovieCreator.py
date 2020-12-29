import csv
import numpy as np
from tifffile import imwrite 
import pandas as pd
from skimage.measure import regionprops
from skimage import measure
from scipy import spatial 
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
    
def CreateTrainingMovies(csv_file, image, segimage, crop_size, TotalCategories, trainlabel, save_dir, gridX = 1, gridY = 1, offset = 0, defname = "" ):

            Path(save_dir).mkdir(exist_ok=True)
            name = 1
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
       if offset > 0:
         shiftLX = [-1.0 * offset, 0] 
         shiftRX = [offset, 0]
         shiftLXY = [-1.0 * offset, -1.0 * offset]
         shiftRXY = [offset, -1.0 * offset]
         shiftDLXY = [-1.0 * offset, offset]
         shiftDRXY = [offset, offset]
         shiftUY = [0, -1.0 * offset]
         shiftDY = [0, offset]
         AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY]

       else:
           
          AllShifts = [shiftNone]


       
       currentsegimage = segimage[time,:].astype('uint16')
       for shift in AllShifts:
           
                defaultX = int(x + shift[0])  
                defaultY = int(y + shift[1])
                
                #Get the closest centroid to the clicked point
                properties = measure.regionprops(currentsegimage, currentsegimage)
                TwoDCoordinates = [(prop.centroid[0], prop.centroid[1]) for prop in properties]
                TwoDtree = spatial.cKDTree(TwoDCoordinates)
                TwoDLocation = (defaultY,defaultX)
                closestpoint = TwoDtree.query(TwoDLocation)
                for prop in properties:
                                   
                                   
                            if int(prop.centroid[0]) == int(TwoDCoordinates[closestpoint[1]][0]) and int(prop.centroid[1]) == int(TwoDCoordinates[closestpoint[1]][1]):
                                                minr, minc, maxr, maxc = prop.bbox
                                                center = prop.centroid
                                                height =  abs(maxc - minc)
                                                width =  abs(maxr - minr)
                                                break
                
                Label = np.zeros([TotalCategories + 7])
                Label[trainlabel] = 1
                #T co ordinate
                Label[TotalCategories + 2] = (sizeTminus) / (sizeTminus + sizeTplus)
                if x + shift[0]> sizeX/2 and y + shift[1] > sizeY/2 and x + shift[0] < image.shape[2] and y + shift[1] < image.shape[1] and time > sizeTminus and time + sizeTplus + 1 < image.shape[0]:
                        crop_Xminus = x + shift[0] - int(ImagesizeX/2)
                        crop_Xplus = x + shift[0] + int(ImagesizeX/2)
                        crop_Yminus = y + shift[1] - int(ImagesizeY/2)
                        crop_Yplus = y + shift[1] + int(ImagesizeY/2)
                       
                        # Cut off the region for training movie creation
                        region =(slice(int(time - sizeTminus),int(time + sizeTplus  + 1)),slice(int(crop_Yminus), int(crop_Yplus)),
                              slice(int(crop_Xminus), int(crop_Xplus)))
                        #Define the movie region volume that was cut
                        crop_image = image[region]     
               
                        #X
                        Label[TotalCategories] =  center[1]/sizeX
                        #Y
                        Label[TotalCategories + 1] = center[0]/sizeY
                        #Height
                        Label[TotalCategories + 3] = height/ImagesizeY
                        #Width
                        Label[TotalCategories + 4] = width/ImagesizeX
               
               
                        #Object confidence is 0 for background label else it is 1
                        if trainlabel > 0:
                            Label[TotalCategories + 5] = 1
                        else:
                            Label[TotalCategories + 5] = 0 
                          
                        Label[TotalCategories + 6] = angle  
                      
                        #Write the image as 32 bit tif file 
                        if(crop_image.shape[0] == sizeTplus + sizeTminus + 1 and crop_image.shape[1]== ImagesizeY and crop_image.shape[2]== ImagesizeX):
                                                                imwrite((save_dir + '/' + name + '.tif'  ) , crop_image.astype('float32'))    
        
          
                       #Write the corresponding csv file of labels 
                        writer = csv.writer(open(save_dir + '/' + (name) + ".csv", "w"))
                        for l in Label : writer.writerow ([l])

       
   
                  
    
def CreateTrainingImages(csv_file, image, segimage, crop_size, TotalCategories, trainlabel, save_dir, gridX = 1, gridY = 1, offset = 0, defname = ""):

            Path(save_dir).mkdir(exist_ok=True)
            name = 1
            dataset = pd.read_csv(csv_file)
            time = dataset[dataset.keys()[0]][1:]
            y = dataset[dataset.keys()[1]][1:]
            x = dataset[dataset.keys()[2]][1:]                        
            
            #Categories + XYHW + Confidence 
            for t in x:
               ImageMaker(time[t], y[t], x[t], image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, str(name), save_dir)    
               name = name + 1



def  ImageMaker(time, y, x, image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, name, save_dir):

       sizeX, sizeY = crop_size
       
       ImagesizeX = sizeX * gridX
       ImagesizeY = sizeY * gridY
       
       shiftNone = [0,0]
       if offset > 0:
         shiftLX = [-1.0 * offset, 0] 
         shiftRX = [offset, 0]
         shiftLXY = [-1.0 * offset, -1.0 * offset]
         shiftRXY = [offset, -1.0 * offset]
         shiftDLXY = [-1.0 * offset, offset]
         shiftDRXY = [offset, offset]
         shiftUY = [0, -1.0 * offset]
         shiftDY = [0, offset]
         AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY]

       else:
           
          AllShifts = [shiftNone]

       
       
       currentsegimage = segimage[time,:].astype('uint16')
       for shift in AllShifts:
           
                defaultX = int(x + shift[0])  
                defaultY = int(y + shift[1])
                #Get the closest centroid to the clicked point
                properties = measure.regionprops(currentsegimage, currentsegimage)
                TwoDCoordinates = [(prop.centroid[0], prop.centroid[1]) for prop in properties]
                TwoDtree = spatial.cKDTree(TwoDCoordinates)
                TwoDLocation = (defaultY,defaultX)
                closestpoint = TwoDtree.query(TwoDLocation)
                for prop in properties:
                                   
                                   
                            if int(prop.centroid[0]) == int(TwoDCoordinates[closestpoint[1]][0]) and int(prop.centroid[1]) == int(TwoDCoordinates[closestpoint[1]][1]):
                                                minr, minc, maxr, maxc = prop.bbox
                                                center = prop.centroid
                                                height =  abs(maxc - minc)
                                                width =  abs(maxr - minr)
                                                break
                
                Label = np.zeros([TotalCategories + 5])
                Label[trainlabel] = 1
                
                if x + shift[0]> sizeX/2 and y + shift[1] > sizeY/2 and x + shift[0] < image.shape[2] and y + shift[1] < image.shape[1]:
                            crop_Xminus = x + shift[0] - int(ImagesizeX/2)
                            crop_Xplus = x + shift[0] + int(ImagesizeX/2)
                            crop_Yminus = y + shift[1] - int(ImagesizeY/2)
                            crop_Yplus = y + shift[1] + int(ImagesizeY/2)
      
                            region =(slice(int(time - 1),int(time)),slice(int(crop_Yminus), int(crop_Yplus)),
                                   slice(int(crop_Xminus), int(crop_Xplus)))
                            crop_image = image[region]      
            
                            Label[TotalCategories] =  center[1]/sizeX
                            Label[TotalCategories + 1] = center[0]/sizeY
                            Label[TotalCategories + 2] = height/ImagesizeY
                            Label[TotalCategories + 3] = width/ImagesizeX
                               
                           
                            #Object confidence is 0 for background label else it is 1
                            if trainlabel > 0:
                              Label[TotalCategories + 4] = 1
                            else:
                              Label[TotalCategories + 4] = 0 
                          
                            if(crop_image.shape[1]== ImagesizeY and crop_image.shape[2]== ImagesizeX):
                                     imwrite((save_dir + '/' + name + '.tif'  ) , crop_image.astype('float32'))  
                           
                            writer = csv.writer(open(save_dir + '/' + (name) + ".csv", "w"))
                            for l in Label : writer.writerow ([l])

       

      


