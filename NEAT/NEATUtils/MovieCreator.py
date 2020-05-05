import sys
sys.path.insert(0,"../NEATUtils")
import csv
import os
from glob import glob
import numpy as np
from tifffile import imread 
from NEATUtils.helpers import save_tiff_imagej_compatible
import math
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
    
    
    
    
    
def CreateMoviesXYTZ(csv_file, image, crop_size, shift,TotalCategories, trainlabel, save_dir, name):
    
       x, y, time, z =   np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)   
       
       MovieMaker(x,y,time,z,image, crop_size, shift,TotalCategories, trainlabel, save_dir, name)
       
       
    
def CreateMoviesTXYZ(csv_file, image, crop_size, shift,TotalCategories, trainlabel, save_dir, name):
    
       time, x, y, z =   np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)   
       
       MovieMaker(x,y,time,z,image, crop_size, shift,TotalCategories, trainlabel, save_dir, name)
              
    
def CreateTMovies(csv_file, image, crop_size, shift,TotalCategories, trainlabel, save_dir, name):
    
       x, y, time =   np.loadtxt(csv_file, delimiter = ',', skiprows = 0, unpack=True)   


       MovieMaker(x,y,time,0,image, crop_size, shift,TotalCategories, trainlabel, save_dir, name)    
    
def CreateMovies(csv_file, image, crop_size, shift,TotalCategories, trainlabel, save_dir, name):
    
       time, z, x, y =   np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)   
       
       
       MovieMaker(x,y,time,z,image, crop_size, shift,TotalCategories, trainlabel, save_dir, name)  
    
    
def ReadCSVImage(csv_file, image, crop_size, save_dir):

        
  time , z, x, y = np.loadtxt(csv_file, delimiter = ',', skiprows = 0, unpack=True)
  sizeX, sizeY, sizeT = crop_size
  print(sizeX,sizeY)
  count = 0
  axes = 'TYX'
  for t in range(0, len(time)):
      
      crop_Xminus = x[t] - int(sizeX/2)
      crop_Xplus = x[t] + int(sizeX/2)
      crop_Yminus = y[t] - int(sizeY/2)
      crop_Yplus = y[t] + int(sizeY/2)
      
      region =(slice(int(time[t] - sizeT - 1),int(time[t] + sizeT)),slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
      print(region, image.shape)
      crop_image = image[region]      

      
      count = count + 1
      
      save_tiff_imagej_compatible((save_dir + 'NoEventHere' + str(count) + '.tif'  ) , crop_image, axes)
      
      print(count, time[t], z[t], x[t], y[t])  

def CreateNoShiftMovies(csv_file, image, crop_size,TotalCategories, trainlabel, save_dir, name):
    
       
       time, z, x, y =   np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)   
       sizeX, sizeY, sizeT = crop_size
       NoShiftsave_dir = save_dir + '/' + 'NoShift'
       Path(NoShiftsave_dir).mkdir(exist_ok = True)
       shiftNone = [0,0] 
       offset = 0
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftNone,NoShiftsave_dir,TotalCategories, trainlabel, 'NoShift', name) 
       
def MovieSaver(x,y,time,image,sizeX,sizeY,sizeT,shift,TotalCategories, trainlabel, name, AllShifts, save_dir):
    
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
    
def MovieMaker(x,y,time,z,image, crop_size, shift,TotalCategories, trainlabel, save_dir, name):
    
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
     


       AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY  ]
       
       MovieSaver(x,y,time,image,sizeX,sizeY,sizeT,shift,TotalCategories, trainlabel, name, AllShifts, save_dir)
       
    

     
def CreateCropMovies(ImageDir, crop_size, shift,TotalCategories, trainlabel, save_dir, name):

    



        Images = sorted(glob(ImageDir + '/' + '*.tif'))
        Images = list(map(imread, Images))
        count = 0
        for i in range(0, len(Images)):

            image = Images[i]
            count = count + 1
            x = image.shape[1]//2
            y = image.shape[2]//2
            time = 4
   
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
            SavePatchImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftNone,NoShiftsave_dir,TotalCategories, trainlabel, 'NoShift',count, name) 
            if shift > 0: 
              Path(N10LXShiftsave_dir).mkdir(exist_ok = True)
              Path(N10RXShiftsave_dir).mkdir(exist_ok = True)
              Path(N10LXYShiftsave_dir).mkdir(exist_ok = True)
              Path(N10RXYShiftsave_dir).mkdir(exist_ok = True)
              Path(N10DLXYShiftsave_dir).mkdir(exist_ok = True)
              Path(N10DRXYShiftsave_dir).mkdir(exist_ok = True)
              Path(N10UYShiftsave_dir).mkdir(exist_ok = True)
              Path(N10DYShiftsave_dir).mkdir(exist_ok = True)
       
              SavePatchImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftLX,N10LXShiftsave_dir,TotalCategories, trainlabel, 'ShiftLX', count, name) 
       
              SavePatchImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftRX,N10RXShiftsave_dir,TotalCategories, trainlabel, 'ShiftRX', count, name) 
       
              SavePatchImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftLXY,N10LXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftLXY',count,  name) 
       
              SavePatchImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftRXY,N10RXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftRXY', count, name) 
       
              SavePatchImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDLXY,N10DLXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDLXY',count,  name) 
       
              SavePatchImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDRXY,N10DRXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDRXY',count,  name) 
       
       
              SavePatchImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftUY,N10UYShiftsave_dir,TotalCategories, trainlabel, 'ShiftUY', count, name) 
       
              SavePatchImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDY,N10DYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDY',count,  name)     
       
def CreateMovieMovies(ImageDir, crop_size, shift,TotalCategories, trainlabel, save_dir, name):

    



        Images = sorted(glob(ImageDir + '/' + '*.tif'))
        Images = list(map(imread, Images))
        count = 0
        for i in range(0, len(Images)):

            image = Images[i]
            count = count + 1
            x = image.shape[1]//2
            y = image.shape[2]//2
            time = 4
   
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
     
            AllShifts = [shiftNone, shiftLX, shiftRX,shiftLXY,shiftRXY,shiftDLXY,shiftDRXY,shiftUY,shiftDY  ]
       
            MovieSaver(x,y,time,image,sizeX,sizeY,sizeT,shift,TotalCategories, trainlabel, name, AllShifts, save_dir)     
    

       

            
        
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
          
  
def CreateCropImageMovies(ImageDir, crop_size, shift,TotalCategories, trainlabel, save_dir, name):

    



        Images = sorted(glob(ImageDir + '/' + '*.tif'))
        Images = list(map(imread, Images))
        count = 0
        for i in range(0, len(Images)):

            image = Images[i]
            count = count + 1
            x = image.shape[1]//2
            y = image.shape[2]//2
            time = 4
   
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
            SavePatchMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftNone,NoShiftsave_dir,TotalCategories, trainlabel, 'NoShift',count, name) 
            if shift > 0:
              Path(N10LXShiftsave_dir).mkdir(exist_ok = True)
              Path(N10RXShiftsave_dir).mkdir(exist_ok = True)
              Path(N10LXYShiftsave_dir).mkdir(exist_ok = True)
              Path(N10RXYShiftsave_dir).mkdir(exist_ok = True)
              Path(N10DLXYShiftsave_dir).mkdir(exist_ok = True)
              Path(N10DRXYShiftsave_dir).mkdir(exist_ok = True)
              Path(N10UYShiftsave_dir).mkdir(exist_ok = True)
              Path(N10DYShiftsave_dir).mkdir(exist_ok = True)
              SavePatchMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftLX,N10LXShiftsave_dir,TotalCategories, trainlabel, 'ShiftLX', count, name) 
       
              SavePatchMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftRX,N10RXShiftsave_dir,TotalCategories, trainlabel, 'ShiftRX', count, name) 
       
              SavePatchMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftLXY,N10LXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftLXY',count,  name) 
       
              SavePatchMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftRXY,N10RXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftRXY', count, name) 
       
              SavePatchMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDLXY,N10DLXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDLXY',count,  name) 
       
              SavePatchMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDRXY,N10DRXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDRXY',count,  name) 
       
       
              SavePatchMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftUY,N10UYShiftsave_dir,TotalCategories, trainlabel, 'ShiftUY', count, name) 
       
              SavePatchMovieImages(x,y,time,image,sizeX,sizeY,sizeT,offset,shiftDY,N10DYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDY',count,  name)      
               
              
def CreateImages(csv_file, image, crop_size, shift,TotalCategories, trainlabel,  save_dir, name):
    
       time, z, x, y =   np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)   
       sizeX, sizeY = crop_size
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
       Save2DImages(x,y,time,image,sizeX,sizeY,offset,shiftNone,NoShiftsave_dir,TotalCategories, trainlabel, 'NoShift',name) 
       if shift > 0:
            
         Path(N10LXShiftsave_dir).mkdir(exist_ok = True)
         Path(N10RXShiftsave_dir).mkdir(exist_ok = True)
         Path(N10LXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10RXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DLXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DRXYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10UYShiftsave_dir).mkdir(exist_ok = True)
         Path(N10DYShiftsave_dir).mkdir(exist_ok = True)
         Save2DImages(x,y,time,image,sizeX,sizeY,offset,shiftLX,N10LXShiftsave_dir,TotalCategories, trainlabel, 'ShiftLX',name) 
       
         Save2DImages(x,y,time,image,sizeX,sizeY,offset,shiftRX,N10RXShiftsave_dir,TotalCategories, trainlabel, 'ShiftRX',name) 
       
         Save2DImages(x,y,time,image,sizeX,sizeY,offset,shiftLXY,N10LXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftLXY',name) 
       
         Save2DImages(x,y,time,image,sizeX,sizeY,offset,shiftRXY,N10RXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftRXY',name) 
       
         Save2DImages(x,y,time,image,sizeX,sizeY,offset,shiftDLXY,N10DLXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDLXY',name) 
       
         Save2DImages(x,y,time,image,sizeX,sizeY,offset,shiftDRXY,N10DRXYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDRXY',name) 
       
       
         Save2DImages(x,y,time,image,sizeX,sizeY,offset,shiftUY,N10UYShiftsave_dir,TotalCategories, trainlabel, 'ShiftUY',name) 
       
         Save2DImages(x,y,time,image,sizeX,sizeY,offset,shiftDY,N10DYShiftsave_dir,TotalCategories, trainlabel, 'ShiftDY',name)       
       


def SavePatchImages(x,y,time, image, sizeX, sizeY, sizeT,offset, shift, savedir, TotalCategories, trainlabel, name, count, appendname):
    
   
   axes = 'TYX'
   Label = np.zeros([TotalCategories + 3])
   Label[trainlabel] = 1
   Label[TotalCategories + 2] = 0.5
   
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


   crop_Xminus = x - shift[0] - int(sizeX/2)
   crop_Xplus = x - shift[0] + int(sizeX/2)
   crop_Yminus = y - shift[1] - int(sizeY/2)
   crop_Yplus = y - shift[1] + int(sizeY/2)
      
   region =(slice(int(time - sizeT - 1),int(time + sizeT )),slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
   crop_image = image[region]      
       
      
   
   if(crop_image.shape[0] == 2 * sizeT + 1 and crop_image.shape[1]== sizeX and crop_image.shape[2]== sizeY):
        save_tiff_imagej_compatible((savedir + '/' + name + str(count) + appendname + '.tif'  ) , crop_image, axes)
   

def SavePatchMovieImages(x,y,time, image, sizeX, sizeY, sizeT,offset, shift, savedir, TotalCategories, trainlabel, name, count, appendname):
    
   
   axes = 'TYX'
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


   crop_Xminus = x - shift[0] - int(sizeX/2)
   crop_Xplus = x - shift[0] + int(sizeX/2)
   crop_Yminus = y - shift[1] - int(sizeY/2)
   crop_Yplus = y - shift[1] + int(sizeY/2)
      
    
       
      
   for i in range(-sizeT - 1, sizeT):
           
           region = (slice(int(time + i),int(time + i + 1 )),slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
           crop_image = image[region]      
       
      
           count = count + 1
              
           if(crop_image.shape[1]== sizeX and crop_image.shape[2]== sizeY):
             save_tiff_imagej_compatible((savedir + '/' + name + str(count) + 'Slice' + str(i) +   appendname + '.tif'  ) , crop_image[0,:], axes)
        
        

def SaveImages(x,y,time, image, sizeX, sizeY, sizeT,offset, shift, savedir, TotalCategories, trainlabel, name, appendname):
    
   count = 0
   axes = 'TYX'
   Label = np.zeros([TotalCategories + 3])
   Label[trainlabel] = 1
   Label[TotalCategories + 2] = 0.5
   
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
      
       region =(slice(int(time[t] - sizeT - 1),int(time[t] + sizeT )),slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
       crop_image = image[region]      
       
      
       count = count + 1
              
       if(crop_image.shape[0] == 2 * sizeT + 1 and crop_image.shape[1]== sizeX and crop_image.shape[2]== sizeY):
        save_tiff_imagej_compatible((savedir + '/' + name + str(count) + appendname + '.tif'  ) , crop_image, axes)
        
    
    
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
             save_tiff_imagej_compatible((savedir + '/' + name + str(count) + 'Slice' + str(i)   + appendname + '.tif'  ) , crop_image[0,:], axes)
        
def SaveMovies(x,y,time, image, sizeX, sizeY, sizeT,offset, shift, savedir, TotalCategories, trainlabel, name, appendname, count):
    
   axes = 'TYX'
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

       

   if x - shift[0]> sizeX/2 and y - shift[0] > sizeY/2 :
       crop_Xminus = x- shift[0] - int(sizeX/2)
       crop_Xplus = x - shift[0] + int(sizeX/2)
       crop_Yminus = y - shift[1] - int(sizeY/2)
       crop_Yplus = y - shift[1] + int(sizeY/2)
       for i in range(-sizeT - 1, sizeT):
           
           region = (slice(0,image.shape[0]),slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
           crop_image = image[region]      
       
           
           
              
           if(crop_image.shape[1]== sizeX and crop_image.shape[2]== sizeY):
             save_tiff_imagej_compatible((savedir + '/' + name + str(count)  + appendname + '.tif'  ) , crop_image, axes)
                
        
    
def Save2DImages(x,y,time, image, sizeX, sizeY,offset, shift, savedir, TotalCategories, trainlabel,  name, appendname):
    
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
      if x[t] - shift[0]> sizeX/2 and y[t] - shift[0] > sizeY/2:
       crop_Xminus = x[t] - shift[0] - int(sizeX/2)
       crop_Xplus = x[t] - shift[0] + int(sizeX/2)
       crop_Yminus = y[t] - shift[1] - int(sizeY/2)
       crop_Yplus = y[t] - shift[1] + int(sizeY/2)
      
       region =(slice(int(time[t]),int(time[t] )),slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
       crop_image = image[region]      
       
      
       count = count + 1
              
       if(crop_image.shape[0] == 1 and crop_image.shape[1]==sizeX and crop_image.shape[2]==sizeY):
        save_tiff_imagej_compatible((savedir + '/' + name + str(count) + appendname + '.tif'  ) , crop_image, axes)
        
    
        
    
  
def main(csv_file, image_file, crop_size,shift, NoShiftsave_dir,N10LXShiftsave_dir,N10RXShiftsave_dir,N10LXYShiftsave_dir,N10RXYShiftsave_dir,N10DLXYShiftsave_dir,N10DRXYShiftsave_dir, N10UYShiftsave_dir,N10DYShiftsave_dir): 
    ReadCSVImage(csv_file, image_file, crop_size, NoShiftsave_dir)

if __name__ == "__main__":
        csv_file = '/Users/aimachine/Documents/VicData/segmentedDiv_Movie2.txt'
        image_file = '/Users/aimachine/Documents/VicData/Movie2.tif'
        NoShiftsave_dir = '/Users/aimachine/NoShiftDivision/'
        
        
        N10LXShiftsave_dir = '/Users/aimachine/10LXShiftDivision/'
        N10RXShiftsave_dir = '/Users/aimachine/10RXShiftDivision/'
        N10LXYShiftsave_dir = '/Users/aimachine/10LXYShiftDivision/'
        N10RXYShiftsave_dir = '/Users/aimachine/10RXYShiftDivision/'
        N10DLXYShiftsave_dir = '/Users/aimachine/10DLXYShiftDivision/'
        N10DRXYShiftsave_dir = '/Users/aimachine/10DRXYShiftDivision/'
        
        N10UYShiftsave_dir = '/Users/aimachine/10UYShiftDivision/'
        N10DYShiftsave_dir = '/Users/aimachine/10DYShiftDivision/'
        
        Path(NoShiftsave_dir).mkdir(exist_ok = True)
        Path(N10LXShiftsave_dir).mkdir(exist_ok = True)
        Path(N10RXShiftsave_dir).mkdir(exist_ok = True)
        Path(N10LXYShiftsave_dir).mkdir(exist_ok = True)
        Path(N10RXYShiftsave_dir).mkdir(exist_ok = True)
        Path(N10DLXYShiftsave_dir).mkdir(exist_ok = True)
        Path(N10DRXYShiftsave_dir).mkdir(exist_ok = True)
        Path(N10UYShiftsave_dir).mkdir(exist_ok = True)
        Path(N10DYShiftsave_dir).mkdir(exist_ok = True)
        
        cropsize = [54,54, 3]
        image= imread(image_file)
        image = np.asarray(image)
        shift = 10
        main(csv_file, image, cropsize,shift, NoShiftsave_dir,N10LXShiftsave_dir,N10RXShiftsave_dir,N10LXYShiftsave_dir,N10RXYShiftsave_dir,N10DLXYShiftsave_dir,N10DRXYShiftsave_dir, N10UYShiftsave_dir,N10DYShiftsave_dir )   
        
         
