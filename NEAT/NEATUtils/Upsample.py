import sys
sys.path.insert(0,"../NEATUtils")
import csv
import os
import cv2
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
    
    
    
def UpsampleMovies(TargetDir, SourceDir, SizeX, SizeY):
    
    
    Raw_path = os.path.join(SourceDir,'*.tif')
    X = glob.glob(Raw_path)

    Path(TargetDir).mkdir(exist_ok = True)
    for image in X:
        
        targetimage = np.zeros([image.shape[0], SizeX, SizeY])
        Name = os.path.basename((os.path.splitext(image)[0])) 
        for i in range(0,image.shape[0]):
            
            targetimage[i,:] = cv2.resize(image[i,:], (SizeX, SizeY), interpolation = cv2.INTER_LINEAR)
        save_tiff_imagej_compatible((TargetDir + '/' + Name + '.tif'  ) , targetimage, 'TYX')

def UpsampleImages(TargetDir, SourceDir, SizeX, SizeY):
    
    
    Raw_path = os.path.join(SourceDir,'*.tif')
    X = glob.glob(Raw_path)

    Path(TargetDir).mkdir(exist_ok = True)
    for image in X:
        
        targetimage = np.zeros([SizeX, SizeY])
        Name = os.path.basename((os.path.splitext(image)[0])) 
        
            
        targetimage = cv2.resize(image, (SizeX, SizeY), interpolation = cv2.INTER_LINEAR)
        save_tiff_imagej_compatible((TargetDir + '/' + Name + '.tif'  ) , targetimage, 'YX')
        
        
        
        
        