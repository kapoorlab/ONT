import csv
from glob import glob
import numpy as np
from tifffile import imread, imwrite 
import pandas as pd
import math
from skimage.measure import regionprops
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
    
    
    
def CreateTrainingMovies(csv_file, image, segimage, crop_size, TotalCategories, trainlabel, save_dir, gridX = 1, gridY = 1, offset = 0 ):

            Path(save_dir).mkdir(exist_ok=True)
            name = 1
            dataset = pd.read_csv(csv_file)
            if len(dataset.keys() >= 3):
                
                time = dataset[dataset.keys()[0]][1:]
                y = dataset[dataset.keys()[1]][1:]
                x = dataset[dataset.keys()[2]][1:]
                angle = np.zeros_like(time)                        
            if len(dataset.keys() > 3):
                
                angle = dataset[dataset.keys()[3]][1:]      
            
            #Categories + XYTHW + Confidence + Angle
            for t in time:
               MovieMaker(time[t], y[t], x[t], angle[t], image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, str(name), save_dir)
               name = name + 1
            
            
    
def CreateTrainingImages(csv_file, image, segimage, crop_size, TotalCategories, trainlabel, save_dir, gridX = 1, gridY = 1, offset = 0):

            Path(save_dir).mkdir(exist_ok=True)
            name = 1
            dataset = pd.read_csv(csv_file)
            y = dataset[dataset.keys()[0]][1:]
            x = dataset[dataset.keys()[1]][1:]                        
            
            #Categories + XYHW + Confidence 
            for t in x:
               ImageMaker(y[t], x[t], image, segimage, crop_size, gridX, gridY, offset, TotalCategories, trainlabel, str(name) save_dir)    
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

       Label = np.zeros([TotalCategories + 7])
       Label[trainlabel] = 1
       #T co ordinate
       Label[TotalCategories + 2] = (sizeTminus) / (sizeTminus + sizeTplus)
       
       currentsegimage = segimage[time,:].astype('uint16')
       for shift in AllShifts:
           
               defaultX = int((sizeX//2 + shift[0])/sizeX)  
               defaultY = int((sizeY//2 + shift[1])/sizeY)
               imageLabel = currentsegimage[defaultY, defaultX]
               for region in regionprops(currentsegimage):
           
                       if region.label == imageLabel and imageLabel > 0:
                                minr, minc, maxr, maxc = region.bbox
              
                                center = region.centroid
                                height =  abs(maxc - minc)
                                width =  abs(maxr - minr)
               

               Label[TotalCategories] =  center[1]/sizeX
               Label[TotalCategories + 1] = center[0]/sizeY
               Label[TotalCategories + 3] = height/ImagesizeY
               Label[TotalCategories + 4] = width/ImagesizeX
               
               
               #Object confidence is 0 for background label else it is 1
               if trainlabel > 0:
                  Label[TotalCategories + 5] = 1
               else:
                  Label[TotalCategories + 5] = 0 
                  
               Label[TotalCategories + 6] = angle    
               writer = csv.writer(open(savedir + '/' + (name) + ".csv", "w"))
           for l in Label : writer.writerow ([l])

       
           if x - shift[0]> sizeX/2 and y - shift[1] > sizeY/2 and time > sizeTminus and time + sizeTplus + 1 < image.shape[0]:
                          crop_Xminus = x[t] + shift[0] - int(ImagesizeX/2)
                          crop_Xplus = x[t] + shift[0] + int(ImagesizeX/2)
                          crop_Yminus = y[t] + shift[1] - int(ImagesizeY/2)
                          crop_Yplus = y[t] + shift[1] + int(ImagesizeY/2)
      
           region =(slice(int(time - sizeTminus - 1),int(time + sizeTplus )),slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
           crop_image = image[region]      
       
      
              
           if(crop_image.shape[0] == sizeTplus + sizeTminus + 1 and crop_image.shape[1]== ImagesizeY and crop_image.shape[2]== ImagesizeX):
                         imwrite((savedir + '/' + name + '.tif'  ) , crop_image.astype('float32'))       
       
def  ImageMaker(y, x, image, segimage, crop_size, gridX, gridY, shift, TotalCategories, trainlabel, name, save_dir):

       sizeX, sizeY = crop_size
       
       ImagesizeX = sizeX * gridX
       ImagesizeY = sizeY * gridY
       
       shiftNone = [0,0] 
       shiftLX = [-1.0 * shift, 0] 
       shiftRX = [shift, 0]
       shiftLXY = [-1.0 * shift, -1.0 * shift]
       shiftRXY = [shift, -1.0 * shift]
       shiftDLXY = [-1.0 * shift,shift]
       shiftDRXY = [shift, shift]
       shiftUY = [0, -1.0 * shift]
       shiftDY = [0, shift]
       AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY]

      




       
def MovieSaver(x,y,time,image,sizeX,sizeY,sizeTplus,sizeTminus, shift, TotalCategories, trainlabel, name, AllShifts, save_dir):
    
       shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY  = AllShifts
       offset = shift      
       
       
       NoShiftsave_dir = save_dir + '/' + 'NoShift'
       N10LXShiftsave_dir = save_dir + '/' + 'ShiftLX'
       N10RXShiftsave_dir = save_dir + '/' + 'ShiftRX'
       N10LXYShiftsave_dir = save_dir + '/' + 'ShiftLXY'
       N10RXYShiftsave_dir = save_dir + '/' + 'ShiftRXY'
       N10DLXYShiftsave_dir = save_dir + '/' + 'ShiftDLXY'
       N10DRXYShiftsave_dir = save_dir + '/' + 'ShiftDRXY'
       N10UYShiftsave_dir = save_dir + '/' + 'ShiftUY'
       N10DYShiftsave_dir = save_dir + '/' + 'ShiftDY'

       Path(NoShiftsave_dir).mkdir(exist_ok = True)
       
       SaveMovies(x,y,time,image,sizeX,sizeY,sizeTplus,sizeTminus,offset,shiftNone,NoShiftsave_dir,TotalCategories, trainlabel, 'NoShift', name) 
       if shift > 0:
         Path(N10LXShiftsave_dir).mkdir(exist_ok = True)
         Path(N10RXShiftsave_dir).mkdir(exist_ok = True)
         Path(N10LXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10RXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DLXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DRXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10UYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DYShiftsave_dir).mkdir(exist_ok = True)
            
         SaveMovies(x,y,time,image,sizeX,sizeY,sizeTplus,sizeTminus,offset,shiftLX,N10LXShiftsave_dir,TotalCategories, trainlabel, 'ShiftLX', name) 
       
         SaveMovies(x,y,time,image,sizeX,sizeY,sizeTplus,sizeTminus,offset,shiftRX,N10RXShiftsave_dir,TotalCategories, trainlabel, 'ShiftRX', name) 
       
         SaveMovies(x,y,time,image,sizeX,sizeY,sizeTplus,sizeTminus,offset,shiftLXY,N10LXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftLXY', name) 
       
         SaveMovies(x,y,time,image,sizeX,sizeY,sizeTplus,sizeTminus,offset,shiftRXY,N10RXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftRXY', name) 
       
         SaveMovies(x,y,time,image,sizeX,sizeY,sizeTplus,sizeTminus,offset,shiftDLXY,N10DLXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDLXY', name) 
       
         SaveMovies(x,y,time,image,sizeX,sizeY,sizeTplus,sizeTminus,offset,shiftDRXY,N10DRXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDRXY', name) 
       
       
         SaveMovies(x,y,time,image,sizeX,sizeY,sizeTplus,sizeTminus,offset,shiftUY,N10UYShiftsave_dir,TotalCategories, trainlabel, 'ShiftUY', name) 
       
         SaveMovies(x,y,time,image,sizeX,sizeY,sizeTplus,sizeTminus,offset,shiftDY,N10DYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDY', name)
   
def ImageSaver(x,y,time,image,sizeX,sizeY,sizeT,shift,TotalCategories, trainlabel, name, AllShifts, save_dir):
    
       shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY  = AllShifts
       offset = shift      
       
       
       NoShiftsave_dir = save_dir + '/' + 'NoShift'
       N10LXShiftsave_dir = save_dir + '/' + 'ShiftLX'
       N10RXShiftsave_dir = save_dir + '/' + 'ShiftRX'
       N10LXYShiftsave_dir = save_dir + '/' + 'ShiftLXY'
       N10RXYShiftsave_dir = save_dir + '/' + 'ShiftRXY'
       N10DLXYShiftsave_dir = save_dir + '/' + 'ShiftDLXY'
       N10DRXYShiftsave_dir = save_dir + '/' + 'ShiftDRXY'
       N10UYShiftsave_dir = save_dir + '/' + 'ShiftUY'
       N10DYShiftsave_dir = save_dir + '/' + 'ShiftDY'

       Path(NoShiftsave_dir).mkdir(exist_ok = True)
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftNone,NoShiftsave_dir,TotalCategories, trainlabel, 'NoShift', name) 
       if shift > 0:
         Path(N10LXShiftsave_dir).mkdir(exist_ok = True)
         Path(N10RXShiftsave_dir).mkdir(exist_ok = True)
         Path(N10LXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10RXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DLXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DRXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10UYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DYShiftsave_dir).mkdir(exist_ok = True)
            
         SaveImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftLX,N10LXShiftsave_dir,TotalCategories, trainlabel, 'ShiftLX', name) 
       
         SaveImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftRX,N10RXShiftsave_dir,TotalCategories, trainlabel, 'ShiftRX', name) 
       
         SaveImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftLXY,N10LXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftLXY', name) 
       
         SaveImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftRXY,N10RXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftRXY', name) 
       
         SaveImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDLXY,N10DLXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDLXY', name) 
       
         SaveImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDRXY,N10DRXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDRXY', name) 
       
       
         SaveImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftUY,N10UYShiftsave_dir,TotalCategories, trainlabel, 'ShiftUY', name) 
       
         SaveImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDY,N10DYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDY', name)


     
        
def CreateImageMovies(csv_file, image, crop_size, shift,TotalCategories, trainlabel, save_dir, name):
    
       time, z, x, y =   np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)   
       sizeX, sizeY, sizeT = crop_size
       shiftNone = [0,0] 
       shiftLX = [-1.0 * shift, 0] 
       shiftRX = [shift, 0]
       shiftLXY = [-1.0 * shift, -1.0 * shift]
       shiftRXY = [shift, -1.0 * shift]
       shiftDLXY = [-1.0 * shift,shift]
       shiftDRXY = [shift, shift]
       shiftUY = [0, -1.0 * shift]
       shiftDY = [0, shift]
     
       NoShiftsave_dir = save_dir + '/' + 'NoShift'
       N10LXShiftsave_dir = save_dir + '/' + 'ShiftLX'
       N10RXShiftsave_dir = save_dir + '/' + 'ShiftRX'
       N10LXYShiftsave_dir = save_dir + '/' + 'ShiftLXY'
       N10RXYShiftsave_dir = save_dir + '/' + 'ShiftRXY'
       N10DLXYShiftsave_dir = save_dir + '/' + 'ShiftDLXY'
       N10DRXYShiftsave_dir = save_dir + '/' + 'ShiftDRXY'
       N10UYShiftsave_dir = save_dir + '/' + 'ShiftUY'
       N10DYShiftsave_dir = save_dir + '/' + 'ShiftDY'

       Path(NoShiftsave_dir).mkdir(exist_ok = True)

       
       
       offset = shift       
       SaveMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftNone,NoShiftsave_dir,TotalCategories, trainlabel, 'NoShift', name) 
       if shift > 0:
         Path(N10LXShiftsave_dir).mkdir(exist_ok = True)
         Path(N10RXShiftsave_dir).mkdir(exist_ok = True)
         Path(N10LXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10RXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DLXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DRXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10UYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DYShiftsave_dir).mkdir(exist_ok = True)
         SaveMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftLX,N10LXShiftsave_dir,TotalCategories, trainlabel, 'ShiftLX', name) 
       
         SaveMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftRX,N10RXShiftsave_dir,TotalCategories, trainlabel, 'ShiftRX', name) 
       
         SaveMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftLXY,N10LXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftLXY', name) 
       
         SaveMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftRXY,N10RXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftRXY', name) 
       
         SaveMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDLXY,N10DLXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDLXY', name) 
       
         SaveMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDRXY,N10DRXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDRXY', name) 
       
       
         SaveMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftUY,N10UYShiftsave_dir,TotalCategories, trainlabel, 'ShiftUY', name) 
       
         SaveMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDY,N10DYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDY', name)
          




def SaveMovies(x,y,time, image, sizeX, sizeY, sizeTplus,sizeTminus,offset, shift, savedir, TotalCategories, trainlabel, name, appendname):
    
    
 
       
    
   count = 0
   axes = 'TYX'
   Label = np.zeros([TotalCategories + 3])
   Label[trainlabel] = 1
   Label[TotalCategories + 2] = (sizeTminus) / (sizeTminus + sizeTplus)
   
   if name == 'NoShift':
       Label[TotalCategories] = 0.5
       Label[TotalCategories + 1] = 0.5
       
   if name == 'ShiftLX':
       Label[TotalCategories] = (sizeX//2 - offset)/sizeX # 0.31
       Label[TotalCategories + 1] = 0.5
       
   if name == 'ShiftRX':
       Label[TotalCategories] = (sizeX//2 + offset)/sizeX # 0.67
       Label[TotalCategories + 1] = 0.5
       
   if name == 'ShiftLXY':
       Label[TotalCategories] = (sizeX//2 - offset)/sizeX # 0.31
       Label[TotalCategories + 1] = (sizeY//2 - offset)/sizeY # 0.31
       
   if name == 'ShiftRXY':
       Label[TotalCategories] = (sizeX//2 + offset)/sizeX # 0.67
       Label[TotalCategories + 1] = (sizeY//2 - offset)/sizeY # 0.31
       
   if name == 'ShiftDLXY':
       Label[TotalCategories] = (sizeX//2 - offset)/sizeX #0.31
       Label[TotalCategories + 1] = (sizeY//2 + offset)/sizeY #0.67
   
   if name == 'ShiftDRXY':
       Label[TotalCategories] = (sizeX//2 + offset)/sizeX  #0.67
       Label[TotalCategories + 1] = (sizeY//2 + offset)/sizeY #0.67
       
   if name == 'ShiftUY':
       Label[TotalCategories] = 0.5
       Label[TotalCategories + 1] = (sizeY//2 - offset)/sizeY # 0.31
       
   if name == 'ShiftDY':
       Label[TotalCategories] = 0.5
       Label[TotalCategories + 1] = (sizeY//2 + offset)/sizeY #0.67
      
       
   writer = csv.writer(open(savedir + '/' + (name) + 'Label'  +".csv", "w"))
   
   for l in Label : writer.writerow ([l])

       
   for t in range(0, len(time)):
      if math.isnan(x[t]): 
         continue 
      if x[t] - shift[0]> sizeX/2 and y[t] - shift[0] > sizeY/2 and time[t] > sizeTminus and time[t] + sizeTplus + 1 < image.shape[0]:
       crop_Xminus = x[t] - shift[0] - int(sizeX/2)
       crop_Xplus = x[t] - shift[0] + int(sizeX/2)
       crop_Yminus = y[t] - shift[1] - int(sizeY/2)
       crop_Yplus = y[t] - shift[1] + int(sizeY/2)
      
       region =(slice(int(time[t] - sizeTminus - 1),int(time[t] + sizeTplus )),slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
       crop_image = image[region]      
       
      
       count = count + 1
              
       if(crop_image.shape[0] == sizeTplus + sizeTminus + 1 and crop_image.shape[1]== sizeX and crop_image.shape[2]== sizeY):
        imwrite((savedir + '/' + name + str(count) + appendname + '.tif'  ) , crop_image.astype('float32'))
        
    
def SaveImages(x,y,time, image, sizeX, sizeY, sizeT,offset, shift, savedir, TotalCategories, trainlabel, name, appendname):
    
   count = 0
   axes = 'YX'
   Label = np.zeros([TotalCategories + 2])
   Label[trainlabel] = 1
   
   if name == 'NoShift':
       Label[TotalCategories] = 0.5
       Label[TotalCategories + 1] = 0.5
       
   if name == 'ShiftLX':
       Label[TotalCategories] = (sizeX//2 - offset)/sizeX # 0.31
       Label[TotalCategories + 1] = 0.5
       
   if name == 'ShiftRX':
       Label[TotalCategories] = (sizeX//2 + offset)/sizeX # 0.67
       Label[TotalCategories + 1] = 0.5
       
   if name == 'ShiftLXY':
       Label[TotalCategories] = (sizeX//2 - offset)/sizeX # 0.31
       Label[TotalCategories + 1] = (sizeY//2 - offset)/sizeY # 0.31
       
   if name == 'ShiftRXY':
       Label[TotalCategories] = (sizeX//2 + offset)/sizeX # 0.67
       Label[TotalCategories + 1] = (sizeY//2 - offset)/sizeY # 0.31
       
   if name == 'ShiftDLXY':
       Label[TotalCategories] = (sizeX//2 - offset)/sizeX #0.31
       Label[TotalCategories + 1] = (sizeY//2 + offset)/sizeY #0.67
   
   if name == 'ShiftDRXY':
       Label[TotalCategories] = (sizeX//2 + offset)/sizeX  #0.67
       Label[TotalCategories + 1] = (sizeY//2 + offset)/sizeY #0.67
       
   if name == 'ShiftUY':
       Label[TotalCategories] = 0.5
       Label[TotalCategories + 1] = (sizeY//2 - offset)/sizeY # 0.31
       
   if name == 'ShiftDY':
       Label[TotalCategories] = 0.5
       Label[TotalCategories + 1] = (sizeY//2 + offset)/sizeY #0.67
      
       
   writer = csv.writer(open(savedir + '/' + (name) + 'Label'  +".csv", "w"))
   
   for l in Label : writer.writerow ([l])

       
   for t in range(0, len(time)):
      if math.isnan(x[t]): 
         continue 
     
      crop_Xminus = x[t] - shift[0] - int(sizeX/2)
      crop_Xplus = x[t] - shift[0] + int(sizeX/2)
      crop_Yminus = y[t] - shift[1] - int(sizeY/2)
      crop_Yplus = y[t] - shift[1] + int(sizeY/2)
      if(time[t] + sizeT - 1 > 0):
       region =(slice(int(time[t] + sizeT - 1),int(time[t] + sizeT )),slice(int(crop_Yminus), int(crop_Yplus)),
                       slice(int(crop_Xminus), int(crop_Xplus)))
       crop_image = image[region]      
       
      
       count = count + 1
       if(crop_image.shape[1]== sizeX and crop_image.shape[2]== sizeY):
        try:
          imwrite((savedir + '/' + name + str(count) + appendname + '.tif'  ) , crop_image.astype('float32'))
        except:
            print('zero array')
      

def SaveMovieImages(x,y,time, image, sizeX, sizeY, sizeT,offset, shift, savedir, TotalCategories, trainlabel, name, appendname):
    
   count = 0
   axes = 'YX'
   Label = np.zeros([TotalCategories + 2])
   Label[trainlabel] = 1
   
   if name == 'NoShift':
       Label[TotalCategories] = 0.5
       Label[TotalCategories + 1] = 0.5
       
   if name == 'ShiftLX':
       Label[TotalCategories] = (sizeX//2 - offset)/sizeX # 0.31
       Label[TotalCategories + 1] = 0.5
       
   if name == 'ShiftRX':
       Label[TotalCategories] = (sizeX//2 + offset)/sizeX # 0.67
       Label[TotalCategories + 1] = 0.5
       
   if name == 'ShiftLXY':
       Label[TotalCategories] = (sizeX//2 - offset)/sizeX # 0.31
       Label[TotalCategories + 1] = (sizeY//2 - offset)/sizeY # 0.31
       
   if name == 'ShiftRXY':
       Label[TotalCategories] = (sizeX//2 + offset)/sizeX # 0.67
       Label[TotalCategories + 1] = (sizeY//2 - offset)/sizeY # 0.31
       
   if name == 'ShiftDLXY':
       Label[TotalCategories] = (sizeX//2 - offset)/sizeX #0.31
       Label[TotalCategories + 1] = (sizeY//2 + offset)/sizeY #0.67
   
   if name == 'ShiftDRXY':
       Label[TotalCategories] = (sizeX//2 + offset)/sizeX  #0.67
       Label[TotalCategories + 1] = (sizeY//2 + offset)/sizeY #0.67
       
   if name == 'ShiftUY':
       Label[TotalCategories] = 0.5
       Label[TotalCategories + 1] = (sizeY//2 - offset)/sizeY # 0.31
       
   if name == 'ShiftDY':
       Label[TotalCategories] = 0.5
       Label[TotalCategories + 1] = (sizeY//2 + offset)/sizeY #0.67
      
       
   writer = csv.writer(open(savedir + '/' + (name) + 'Label'  +".csv", "w"))
   
   for l in Label : writer.writerow ([l])

       
   for t in range(0, len(time)):
      if math.isnan(x[t]): 
         continue 
      if x[t] - shift[0]> sizeX/2 and y[t] - shift[0] > sizeY/2 and time[t] > sizeT and time[t] + sizeT + 1 < image.shape[0]:
       crop_Xminus = x[t] - shift[0] - int(sizeX/2)
       crop_Xplus = x[t] - shift[0] + int(sizeX/2)
       crop_Yminus = y[t] - shift[1] - int(sizeY/2)
       crop_Yplus = y[t] - shift[1] + int(sizeY/2)
       for i in range(-sizeT - 1, sizeT):
           
           region = (slice(int(time[t] + i),int(time[t] + i + 1 )),slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
           crop_image = image[region]      
       
           
           count = count + 1
              
           if(crop_image.shape[1]== sizeX and crop_image.shape[2]== sizeY):
               imwrite((savedir + '/' + name + str(count) + 'Slice' + str(i)   + appendname + '.tif'  ) , crop_image[0,:].astype('float32'))
        

 