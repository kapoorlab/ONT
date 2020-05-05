import sys
sys.path.insert(0,"../NEATUtils")

import numpy as np
from tifffile import imread 
from helpers import save_tiff_imagej_compatible
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


def CreateMovies(csv_file, image, crop_size, shift, save_dir):
    
       x,y,time, indicator =   np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)   
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
       Path(N10LXShiftsave_dir).mkdir(exist_ok = True)
       Path(N10RXShiftsave_dir).mkdir(exist_ok = True)
       Path(N10LXYShiftsave_dir).mkdir(exist_ok = True)
       Path(N10RXYShiftsave_dir).mkdir(exist_ok = True)
       Path(N10DLXYShiftsave_dir).mkdir(exist_ok = True)
       Path(N10DRXYShiftsave_dir).mkdir(exist_ok = True)
       Path(N10UYShiftsave_dir).mkdir(exist_ok = True)
       Path(N10DYShiftsave_dir).mkdir(exist_ok = True)
       
       
              
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftNone,NoShiftsave_dir,'NoShift') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftLX,N10LXShiftsave_dir,'ShiftLX') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftRX,N10RXShiftsave_dir,'ShiftRX') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftLXY,N10LXYShiftsave_dir,'ShiftLXY') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftRXY,N10RXYShiftsave_dir,'ShiftRXY') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftDLXY,N10DLXYShiftsave_dir,'ShiftDLXY') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftDRXY,N10DRXYShiftsave_dir,'ShiftDRXY') 
       
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftUY,N10UYShiftsave_dir,'ShiftUY') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftDY,N10DYShiftsave_dir,'ShiftDY')
       

def VicDot(csv_file, image, crop_size, shift, NoShiftsave_dir, N10LXShiftsave_dir, N10RXShiftsave_dir, N10LXYShiftsave_dir, N10RXYShiftsave_dir, N10DLXYShiftsave_dir, N10DRXYShiftsave_dir, N10UYShiftsave_dir,N10DYShiftsave_dir):

       x,y,time, indicator =   np.loadtxt(csv_file, delimiter = ',', skiprows = 1, unpack=True)   
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
       
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftNone,NoShiftsave_dir,'NoShift') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftLX,N10LXShiftsave_dir,'ShiftLX') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftRX,N10RXShiftsave_dir,'ShiftRX') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftLXY,N10LXYShiftsave_dir,'ShiftLXY') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftRXY,N10RXYShiftsave_dir,'ShiftRXY') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftDLXY,N10DLXYShiftsave_dir,'ShiftDLXY') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftDRXY,N10DRXYShiftsave_dir,'ShiftDRXY') 
       
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftUY,N10UYShiftsave_dir,'ShiftUY') 
       
       SaveImages(x,y,time,image,sizeX,sizeY,sizeT,shiftDY,N10DYShiftsave_dir,'ShiftDY')
       
       

        



def SaveImages(x,y,time, image, sizeX, sizeY, sizeT, shift, savedir, name):
    
   count = 0
   axes = 'TYX'
   for t in range(0, len(time)):
      if math.isnan(x[t]): 
         continue 
      if x[t] - shift[0]> sizeX/2 and y[t] - shift[0] > sizeY/2 and time[t] > sizeT and time[t] + sizeT + 1 < 180:
       crop_Xminus = x[t] - shift[0] - int(sizeX/2)
       crop_Xplus = x[t] - shift[0] + int(sizeX/2)
       crop_Yminus = y[t] - shift[1] - int(sizeY/2)
       crop_Yplus = y[t] - shift[1] + int(sizeY/2)
      
       region =(slice(int(time[t] - sizeT - 1),int(time[t] + sizeT )),slice(int(crop_Yminus), int(crop_Yplus)),
                      slice(int(crop_Xminus), int(crop_Xplus)))
       crop_image = image[region]      
       
      
       count = count + 1
              
       if(crop_image.shape[0] == 7 and crop_image.shape[1]==54 and crop_image.shape[2]==54):
        save_tiff_imagej_compatible((savedir + name + str(count) + '.tif'  ) , crop_image, axes)
        
    
    
    
    
  
def main(csv_file, image_file, crop_size,shift, NoShiftsave_dir,N10LXShiftsave_dir,N10RXShiftsave_dir,N10LXYShiftsave_dir,N10RXYShiftsave_dir,N10DLXYShiftsave_dir,N10DRXYShiftsave_dir, N10UYShiftsave_dir,N10DYShiftsave_dir): 
    #ReadCSVImage(csv_file, image_file, crop_size, NoShiftsave_dir)
    VicDot(csv_file, image, crop_size,shift, NoShiftsave_dir,N10LXShiftsave_dir,N10RXShiftsave_dir,N10LXYShiftsave_dir,N10RXYShiftsave_dir,N10DLXYShiftsave_dir,N10DRXYShiftsave_dir, N10UYShiftsave_dir,N10DYShiftsave_dir)

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
        
         
