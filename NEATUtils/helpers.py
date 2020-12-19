from __future__ import print_function, unicode_literals, absolute_import, division
#import matplotlib.pyplot as plt
import numpy as np
import os
import collections
import csv
import json
import cv2
import glob
from sklearn.model_selection import train_test_split
from tifffile import imread, imwrite
from skimage import measure
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

"""    
    
def MarkerToCSV(MarkerImage):
    
    MarkerImage = MarkerImage.astype('uint16')
    MarkerList = []
    for i in range(0, MarkerImage.shape[0]):
          waterproperties = measure.regionprops(MarkerImage, MarkerImage)
          indices = [prop.centroid for prop in waterproperties]
          MarkerList.append([i, indices[0], indices[1]])
    return  MarkerList
    
  

    
def load_full_training_data(directory, categories, nb_boxes):
    
     #Generate npz file with data and label attributes   
     Raw_path = os.path.join(directory, '*tif')
     X = glob.glob(Raw_path)
     
     Xtrain = []
     Ytrain = []
     
     for fname in X:
         
         image = imread(fname)
         
         Name = os.path.basename(os.path.splitext(fname)[0])
    
         csvfile = directory + Name + '.csv'
         try:
             
             y_t = []
             reader = csv.reader(csvfile, delimiter = ',')
         
             for train_vec in reader:
                   catarr = [float(s) for s in train_vec[0:categories]]
                   xarr = [float(s) for s in train_vec[categories:]]
                   newxarr = []
                   for b in range(nb_boxes):
                       newxarr+= [xarr[s] for s in range(len(xarr))]
                       
                   trainarr = catarr + newxarr    
                   y_t.append(trainarr)
             
                
             Ytrain.append(y_t)     
             Xtrain.append(image)
    
         except:
             
             print('Matching label not found for:', Name ,  '.tif')
    
     Xtrain = np.array(Xtrain)
     Ytrain = np.array(Ytrain)
  

     print('number of  images:\t', Xtrain.shape[0])
     print('image size (%dD):\t\t',Xtrain.shape)
     print('Labels:\t\t\t\t', Ytrain.shape)
     traindata, validdata, trainlabel, validlabel = train_test_split(Xtrain, Ytrain, train_size=0.95,test_size=0.05, shuffle= True)
     
     return (traindata,trainlabel) , (validdata, validlabel)
        
    
  
    
def time_pad(image, TimeFrames):

         time = image.shape[0]
         
         timeextend = time
         
         while timeextend%TimeFrames!=0:
              timeextend = timeextend + 1
              
         extendimage = np.zeros([timeextend, image.shape[1], image.shape[2]])
              
         extendimage[0:time,:,:] = image
              
         return extendimage      
    
def chunk_list(image, patchshape, stride, pair):
            rowstart = pair[0]
            colstart = pair[1]
            
            endrow = rowstart + patchshape[0]
            endcol = colstart + patchshape[1]
            
            if endrow > image.shape[1]:
                endrow = image.shape[1]
            if endcol > image.shape[2]:
                endcol = image.shape[2]    
            
            
            region = (slice(0,image.shape[0]),slice(rowstart, endrow),
                      slice(colstart, endcol))
            
            # The actual pixels in that region.
            patch = image[region]
                
            # Always normalize patch that goes into the netowrk for getting a prediction score 
            patch = normalizeFloatZeroOne(patch,1,99.8)
            patch = zero_pad(patch, stride,stride)
         
        
            return patch, rowstart, colstart     
           

              
                
def load_json(fpath):
    with open(fpath, 'r') as f:
        return json.load(f)
    
def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        f.write(json.dumps(data,**kwargs))    


   ##CARE csbdeep modification of implemented function
def normalizeFloat(x, pmin = 3, pmax = 99.8, axis = None, eps = 1e-20, dtype = np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis = axis, keepdims = True)
    ma = np.percentile(x, pmax, axis = axis, keepdims = True)
    return normalize_mi_ma(x, mi, ma, eps = eps, dtype = dtype)

def normalizeFloatZeroOne(x, pmin = 3, pmax = 99.8, axis = None, eps = 1e-20, dtype = np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis = axis, keepdims = True)
    ma = np.percentile(x, pmax, axis = axis, keepdims = True)
    return normalizer(x, mi, ma, eps = eps, dtype = dtype)




def normalize_mi_ma(x, mi , ma, eps = 1e-20, dtype = np.float32):
    
    
    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """
    
    
    if dtype is not None:
        x = x.astype(dtype, copy = False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy = False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy = False)
        eps = dtype(eps)
        
    try: 
        import numexpr
        x = numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

        
    return x

def normalizer(x, mi , ma, eps = 1e-20, dtype = np.float32):


    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """


    if dtype is not None:
        x = x.astype(dtype, copy = False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy = False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy = False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

        x = normalizeZeroOne(x)
    return x

def normalizeZeroOne(x):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x
    
def _raise(e):
    raise e

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)



def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes
def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedt      
def X_right_prediction(image,sY, sX, time_prediction, stride, inputtime, Categories_Name, Categories_event_threshold, TrainshapeX, TrainshapeY, TimeFrames):
    
                         LocationBoxes = []
                         j = 0
                         k = 1
                         while True:
                                      j = j + 1
                                      if j > time_prediction.shape[1]:
                                           j = 1
                                           k = k + 1
                             
                                      if k > time_prediction.shape[0]:
                                          break;
                                      
                                      y = (k - 1) * stride
                                      x = (j - 1) * stride
                                      prediction_vector = time_prediction[k-1,j-1,:]
                                      #Note to self k,1 is x j,0 is y 
                                      for p in range(1, len(Categories_Name)):
                                           if prediction_vector[p] > (Categories_event_threshold[p]):
                                                 
                                                 Xbox =  x + sX + prediction_vector[len(Categories_Name)] * TrainshapeX
                                                 Ybox =  y + sY + prediction_vector[len(Categories_Name) + 1] * TrainshapeY
                                                 Tbox = int(inputtime + 1 + prediction_vector[len(Categories_Name) + 2] * TimeFrames)
                                                 
                                                 Score = prediction_vector[p]
                                                 Traw = prediction_vector[len(Categories_Name) + 2]
                                                 Name, Label = Categories_Name[p] 
                                                 box = (x, y,x + TrainshapeX, y + TrainshapeY, Xbox,Ybox, Score, Tbox, Label,Traw )
                                                 
                                                 boxregion = (slice(0,image.shape[0]),slice(y  , y   + TrainshapeY,),slice(x  , x   + TrainshapeX))
                                                 sliceboxregion = image[boxregion]
                                                 try:
                                                  if np.mean(sliceboxregion) > 0.3:
                                                      LocationBoxes.append([box, Label])
                                                 except ValueError:
                                                     print('No box', x, y)
                                                    
                                                    
                         return LocationBoxes      


def draw_labelimages(image, location, time, timelocation ):

     cv2.circle(image, location, 2,(255,0,0), thickness = -1 )

     return image 

def zero_pad(image, TrainshapeX, TrainshapeY):

          time = image.shape[0]
          sizeY = image.shape[2]
          sizeX = image.shape[1]
          
          sizeXextend = sizeX
          sizeYextend = sizeY
         
 
          while sizeXextend%TrainshapeX!=0:
              sizeXextend = sizeXextend + 1
        
          while sizeYextend%TrainshapeY!=0:
              sizeYextend = sizeYextend + 1

          extendimage = np.zeros([time, sizeXextend, sizeYextend])
          
          extendimage[0:time, 0:sizeX, 0:sizeY] = image
              
              
          return extendimage
      
def extra_pad(image, patchX, patchY):

          extendimage = np.zeros([image.shape[0],image.shape[1] + patchX, image.shape[2] + patchY])
          
          extendimage[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] = image     
          
          return extendimage
 
def save_labelimages(save_dir, image, fname, Name):

    
             imwrite((save_dir + Name + os.path.basename(fname) ) , image.astype('uint16'))
        
    
                
def save_csv(save_dir, Event_Count, Name):
      
    Event_data= []

    Path(save_dir).mkdir(exist_ok = True)

    for line in Event_Count:
      Event_data.append(line)
    writer = csv.writer(open(save_dir + "/" + (Name )  +".csv", "w"))
    writer.writerows(Event_data)                
                
