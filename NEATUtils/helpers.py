import numpy as np
import os
import collections
import csv
import json
import math
import cv2
import glob
from scipy import spatial
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tifffile import imread, imwrite
from skimage.segmentation import watershed
from skimage import morphology
from skimage.filters import sobel
from skimage import measure
from pathlib import Path
from skimage.util import invert as invertimage
from scipy.ndimage.morphology import  binary_dilation    
from skimage.measure import label
from skimage.morphology import erosion, dilation, square
import pandas as pd
"""
 @author: Varun Kapoor

"""    
    
"""
This method is used to convert Marker image to a list containing the XY indices for all time points
"""

def remove_big_objects(ar, max_size=6400, connectivity=1, in_place=False):
    
    out = ar.copy()
    ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")



    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out

def IntergerLabelGen(fname, savedir):
            
            BinaryImage = imread(fname)
            Name = os.path.basename(os.path.splitext(fname)[0])
            InputBinaryImage = BinaryImage.astype('uint8')
            
            IntegerImage = np.zeros([BinaryImage.shape[0],BinaryImage.shape[1], BinaryImage.shape[2]])
            for i in tqdm(range(0, InputBinaryImage.shape[0])):
                 
                    BinaryImageOriginal = InputBinaryImage[i,:]
                    Orig = normalizeFloatZeroOne(BinaryImageOriginal)
                    InvertedBinaryImage = invertimage(BinaryImageOriginal)
                    BinaryImage = normalizeFloatZeroOne(InvertedBinaryImage)
                    image = binary_dilation(BinaryImage, iterations = 2)
                    image = invertimage(image)
                    labelclean = label(image)
                    labelclean = remove_big_objects(labelclean, max_size = 15000) 
                    AugmentedLabel = dilation(labelclean, selem = square(3) )
                    AugmentedLabel = np.multiply(AugmentedLabel ,  Orig)
                    IntegerImage[i,:] = AugmentedLabel
            
            imwrite(savedir + Name + '.tif', IntegerImage.astype('uint16'))
            

def MarkerToCSV(MarkerImage):
    
    MarkerImage = MarkerImage.astype('uint16')
    MarkerList = []
    print('Obtaining co-ordinates of markers in all regions')
    for i in range(0, MarkerImage.shape[0]):
          waterproperties = measure.regionprops(MarkerImage, MarkerImage)
          indices = [prop.centroid for prop in waterproperties]
          MarkerList.append([i, indices[0], indices[1]])
    return  MarkerList
    
  
"""
This method is used to import tif files and csv files of the same name as the tif files to create training and validation datasets for training ONEAT network
"""
    

    
class BestAnchorBoxFinder(object):
    def __init__(self, ANCHORS):
        '''
        ANCHORS: a np.array of even number length e.g.
        
        _ANCHORS = [4,2, ##  width=4, height=2,  flat large anchor box
                    2,4, ##  width=2, height=4,  tall large anchor box
                    1,1] ##  width=1, height=1,  small anchor box
        '''
        self.anchors = [BoundBox(0, 0, ANCHORS[2*i], ANCHORS[2*i+1]) 
                        for i in range(int(len(ANCHORS)//2))]
        
    def _interval_overlap(self,interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                 return 0
            else:
                return min(x2,x4) - x3  

    def bbox_iou(self,box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

        union = w1*h1 + w2*h2 - intersect

        return float(intersect) / union
    
    def find(self,center_w, center_h):
        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou     = -1
        # each Anchor box is specialized to have a certain shape.
        # e.g., flat large rectangle, or small square
        shifted_box = BoundBox(0, 0,center_w, center_h)
        ##  For given object, find the best anchor box!
        for i in range(len(self.anchors)): ## run through each anchor box
            anchor = self.anchors[i]
            iou    = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou     = iou
        return(best_anchor,max_iou)    
    
    
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None,classes=None):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax
        ## the code below are used during inference
        # probability
        self.confidence      = confidence
        # class probaiblities [c1, c2, .. cNclass]
        self.set_class(classes)
        
    def set_class(self,classes):
        self.classes = classes
        self.label   = np.argmax(self.classes) 
        
    def get_label(self):  
        return(self.label)
    
    def get_score(self):
        return(self.classes[self.label])
        
        
def load_full_training_data(directory, categories, box_vector, train_image_size, gridX, gridY, nboxes):
    
     #Generate npz file with data and label attributes   
     Raw_path = os.path.join(directory, '*tif')
     X = glob.glob(Raw_path)
     Xtrain = []
     Ytrain = []
     for fname in X:
         
             image = imread(fname)[0,:]
             
             image = np.expand_dims(image, axis=-1)
             
             Name = os.path.basename(os.path.splitext(fname)[0])

             csvfname = directory + Name + '.csv'
             y_t = []
             catarr = []
             trainarr = [] 
             with open(csvfname) as csvfile:
                 reader = csv.reader(csvfile, delimiter = ',')
                 for train_vec in reader:
                         
                         catarr =  [float(s) for s in train_vec[0:categories]]
                         xarr = [float(s) for s in train_vec[categories:categories + box_vector - 1]]
                         newxarr = []
                        
                         for b in range(nboxes):
                               newxarr+= [xarr[s] for s in range(len(xarr))]

                         trainarr = catarr + xarr 
                         
                         
                         y_t.append(trainarr) 
             if len(y_t) > 1:               
              y_t = y_t[:-1]               
                
             Ytrain.append(y_t)     
             Xtrain.append(image)

    
     Xtrain = np.array(Xtrain)
     Ytrain = np.array(Ytrain)
     Ytrain = np.expand_dims(Ytrain, axis=1)
    
     print('number of  images:\t', Xtrain.shape[0])
     print('image size:\t\t',Xtrain.shape)
     print('Labels:\t\t\t\t', Ytrain.shape)
     traindata, validdata, trainlabel, validlabel = train_test_split(Xtrain, Ytrain, train_size=0.95,test_size=0.05, shuffle= True)
     
     return (traindata,trainlabel) , (validdata, validlabel)
        
"""
This method decides if a region should be downsampled for applying the prediction by counting the density around the marker image,
default density veto is 10 cells, if density of region is below this veto the region would be downsampled for applying the ONEAT prediction
"""    
  
def DensityCounter(MarkerImage, TrainshapeX, TrainshapeY, densityveto = 10):

        
    AllDensity = {}

    for i in tqdm(range(0, MarkerImage.shape[0])):
            density = []
            location = []
            currentimage = MarkerImage[i, :].astype('uint16')
            waterproperties = measure.regionprops(currentimage, currentimage)
            indices = [prop.centroid for prop in waterproperties]
            
            for y,x in indices:
                
                           crop_Xminus = x - int(TrainshapeX/2)
                           crop_Xplus = x  + int(TrainshapeX/2)
                           crop_Yminus = y  - int(TrainshapeY/2)
                           crop_Yplus = y  + int(TrainshapeY/2)
                      
                           region =(slice(int(crop_Yminus), int(crop_Yplus)),
                                      slice(int(crop_Xminus), int(crop_Xplus)))
                           crop_image = currentimage[region].astype('uint16')
                           if crop_image.shape[0] >= TrainshapeY and crop_image.shape[1] >= TrainshapeX:
                                    
                                     waterproperties = measure.regionprops(crop_image, crop_image)
                                     
                                     labels = [prop.label for prop in waterproperties]
                                     labels = np.asarray(labels)
                                     #These regions should be downsampled                               
                                     if labels.shape[0] < densityveto:
                                         density.append(labels.shape[0])
                                         location.append((int(y),int(x)))
            #Create a list of TYX marker locations that should be downsampled                             
            AllDensity[str(i)] = [density, location]
    
    return AllDensity

"""
This method takes the integer labelled segmentation image as input and creates a dictionary of markers at all timepoints for easy search
"""    
def MakeTrees(segimage):
    
        AllTrees = {}
        print("Creating Dictionary of marker location for fast search")
        for i in tqdm(range(0, segimage.shape[0])):
                currentimage = segimage[i, :].astype('uint16')
                waterproperties = measure.regionprops(currentimage, currentimage)
                indices = [prop.centroid for prop in waterproperties] 
                if len(indices) > 0:
                    tree = spatial.cKDTree(indices)
                
                    AllTrees[str(i)] =  [tree, indices]
                    
                    
                           
        return AllTrees
    
"""
This method is used to create a segmentation image of an input image (StarDist probability or distance map) using marker controlled watershedding using a mask image (UNET) 
"""    
def WatershedwithMask(Image, Label,mask, grid):
    
    
   
    properties = measure.regionprops(Label, Image)
    Coordinates = [prop.centroid for prop in properties] 
    Coordinates = sorted(Coordinates , key=lambda k: [k[1], k[0]])
    Coordinates.append((0,0))
    Coordinates = np.asarray(Coordinates)
    
    

    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros_like(Image)  
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
    
    markers = morphology.dilation(markers_raw, morphology.disk(2))
    Image = sobel(Image)
    watershedImage = watershed(Image, markers, mask = mask)
    
    return watershedImage, markers     
   
"""
Prediction function for whole image/tile, output is Prediction vector for each image patch it passes over
"""    

def Yoloprediction(image,sy, sx, time_prediction, stride, inputtime, config, key_categories, nboxes, mode, event_type):
    
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

                                      
                                      Classybox, MaxProbLabel = PredictionLoop(j, k, sx, sy, stride, time_prediction, config, key_categories, inputtime, mode, event_type)
                                      #Append the box and the maximum likelehood detected class
                                      LocationBoxes.append([Classybox, MaxProbLabel])         
                             return LocationBoxes
                         
                            
def PredictionLoop(j, k, sx, sy, nboxes, stride, time_prediction, config, key_categories, inputtime, mode, event_type):

                                          total_classes = len(key_categories) 
                                          y = (k - 1) * stride
                                          x = (j - 1) * stride
                                          prediction_vector = time_prediction[k-1,j-1,:]
                                          
                                          xstart = x + sx
                                          ystart = y + sy
                                          Class = {}
                                          #Compute the probability of each class
                                          for (event_name,event_label) in key_categories.items():
                                              
                                              Class[event_name] = prediction_vector[event_label]
                                          xcentermean = 0
                                          ycentermean = 0
                                          widthmean = 0
                                          heightmean = 0
                                          
                                          
                                          trainshapex = config['imagex']
                                          trainshapey = config['imagey']
                                          
                                          for b in nboxes:
                                                  xcenter = xstart + prediction_vector[total_classes + config['x'] ] * trainshapex
                                                  ycenter = ystart + prediction_vector[total_classes + config['y'] ] * trainshapey
                                                  height = prediction_vector[total_classes + config['h']] * trainshapex  
                                                  width = prediction_vector[total_classes + config['w']] * trainshapey
                                                  if config['yolo_v0'] == False:
                                                           confidence = prediction_vector[total_classes + config['c']]
                                                  else:
                                                           confidence = 1         
                                                           #Ignore Yolo boxes with lower than 0.5 confidence
                                                  if config['yolo_v0'] == False and confidence < 0.5:
                                                           continue
                                                  xcentermean = xcentermean + xcenter
                                                  ycentermean = ycentermean + ycenter
                                                  heightmean = heightmean + height
                                                  widthmean = widthmean + width
                                         
                                          
                                          xcentermean = xcentermean/nboxes
                                          ycentermean = ycentermean/nboxes
                                          heightmean = heightmean/nboxes
                                          widthmean = widthmean/nboxes
                                          max_prob_label = np.argmax(prediction_vector[:total_classes])
                                          max_prob_class = prediction_vector[max_prob_label]
                                          
                                          if event_type == 'Dynamic':
                                                  if mode == 'Detection':
                                                          time_frames = config['size_tminus'] + config['size_tplus'] + 1
                                                          real_time_event = int(inputtime + prediction_vector[total_classes + config['t']] * time_frames)
                                                          box_time_event = prediction_vector[total_classes + config['t']]    
                                                  if mode == 'Prediction':
                                                          real_time_event = int(inputtime)
                                                          box_time_event = int(inputtime)
                                                  realangle = math.pi * (prediction_vector[total_classes + config['angle']] - 0.5)
                                                  rawangle = prediction_vector[total_classes + config['angle']]
                                          
                                          if event_type == 'Static':
                                                          real_time_event = int(inputtime)
                                                          box_time_event = int(inputtime)
                                                          realangle = 0
                                                          rawangle = 0
                                          #Compute the box vectors 
                                          box = {'xstart' : xstart, 'ystart' : ystart, 'xcenter' : xcenter, 'ycenter' : ycenter, 'real_time_event' : real_time_event, 'box_time_event' : box_time_event,
                                                 'height' : height, 'width' : width, 'confidence' : confidence, 'realangle' : realangle, 'rawangle' : rawangle}
                                          
                                          
                                          
                                          #Make a single dict object containing the class and the box vectors return also the max prob label
                                          Classybox = {}
                                          for d in [Class,box]:
                                              Classybox.update(d) 
                                          
                                          return Classybox, max_prob_label, max_prob_class
   
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##Need new method for the function below    
    
    
    
    
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
                
