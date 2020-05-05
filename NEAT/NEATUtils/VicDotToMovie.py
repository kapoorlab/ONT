#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:57:47 2019

@author: aimachine
"""

import sys
sys.path.insert(0,"../NEATUtils")
import cv2
import numpy as np
from tifffile import imread 
import matplotlib.pyplot as plt
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
    
def ReadCSVImage(csv_file, image, Label, save_dir):

  x, y, time, indicator =   np.loadtxt(csv_file, delimiter = ',', skiprows = 0, unpack=True)   
  axes = 'TYX'
  ReturnImage = image
  for t in range(0, len(time)):
      
      Currentimage = image[int(time[t])-1, :, :]
      if math.isnan(x[t]):
         continue 
      else:
         location = (int(x[t]), int(y[t]))
         cv2.circle(Currentimage, location, 2,(255,0,0), thickness = -1 )
      ReturnImage[int(time[t])-1, :, :] = Currentimage
         
  save_tiff_imagej_compatible((save_dir + 'VicDots' + Label + '.tif'  ) , ReturnImage, axes)
      
         
def main(csv_file, image_file, crop_size, save_dir): 
    ReadCSVImage(csv_file, image_file, crop_size, save_dir)
    

if __name__ == "__main__":
        csv_file = '/Users/aimachine/Documents/VicData/segmentedDel_Movie2.txt'
        image_file = '/Users/aimachine/Documents/VicData/Movie2.tif'
        save_dir = '/Users/aimachine/Documents/VicData/'
        
        Path(save_dir).mkdir(exist_ok = True)
        image= imread(image_file)
        image = np.asarray(image)
        emptyimage = np.zeros([image.shape[0], image.shape[1], image.shape[2]])
        main(csv_file, emptyimage, 'Apoptosis', save_dir)   
        
         