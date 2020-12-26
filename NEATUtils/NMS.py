import os
import numpy as np
import csv

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




def drawimage(eventlist, basedirResults, fname, Label, EventType):
    
    
   if eventlist is not None: 
    
      

     if EventType == 'Static':
          
      LocationXlist = []
      LocationYlist = []
      Scorelist = []
      Timelist = []   
      Sizelist = []
      
      for i in range(0, len(eventlist)):
                    location, time, Name, Score, size = eventlist[i]
                   
                    returntime = int(time) 

                    LocationXlist.append(location[0])
                    LocationYlist.append(location[1])
                    Timelist.append(returntime)
                    Scorelist.append(Score)
                    Sizelist.append(size)

      Event_Count = np.column_stack([Timelist,LocationYlist, LocationXlist,Scorelist, Sizelist]) 
      Event_data = []
      for line in Event_Count:
        Event_data.append(line)
        writer = csv.writer(open(basedirResults + "/" + str(Label) + "Location" + (os.path.splitext(os.path.basename(fname))[0])  +".csv", "a"))
        writer.writerows(Event_data)
        Event_data = []
        
     if EventType == 'Dynamic':
          
      StaticLocationXlist = []
      StaticLocationYlist = []
      Scorelist = []
      Timelist = []  
      Sizelist = []  
    
    
                          
      eventlist = sorted(eventlist, key = lambda x:x[3], reverse = True) 
      
      for i in range(0, len(eventlist)):
                     Alllocation, time, Name, Score, size = eventlist[i]      
                     returntime = int(time) 

                     StaticLocationXlist.append(Alllocation[0])
                     StaticLocationYlist.append(Alllocation[1])
                     Timelist.append(returntime)
                     Scorelist.append(Score)
                     Sizelist.append(size)
          
 
      Event_Count = np.column_stack([Timelist,StaticLocationYlist, StaticLocationXlist, Scorelist, Sizelist])
      
      Microscope_Event_Count = np.column_stack([StaticLocationXlist, StaticLocationYlist])
      Microscope_Event_Count = np.unique(Microscope_Event_Count, axis=0)
      Event_data = []
      for line in Event_Count:
        Event_data.append(line)
        writer = csv.writer(open(basedirResults + "/" + str(Label) + "Location" + (os.path.splitext(os.path.basename(fname))[0])  +".csv", "a"))
        writer.writerows(Event_data)
        Event_data = [] 
      
      nbPredictions = -1
      for line in  Microscope_Event_Count:
        
        nbPredictions = nbPredictions  + 1
      
      with open(basedirResults + "/" + str(Label) +".ini", "w") as writer:
           writer.write('[main]\n')  
           writer.write('nbPredictions='+str(nbPredictions)+'\n')
           Live_Event_data = []
           count = 1
           for line in Microscope_Event_Count:
              if count > 1:
                  Live_Event_data.append(line)
                  writer.write('['+str(count - 1)+']'+'\n')
                  writer.write('x='+str(Live_Event_data[0][0])+'\n')
                  writer.write('y='+str(Live_Event_data[0][1])+'\n')
                  Live_Event_data = []  
              count = count + 1
              
      

     
      
      
         

            
def StaticEvents( boxes, DownsampleFactor ):
    
    EventList,EventBoxes = NMSSpaceST(boxes, DownsampleFactor)

    return  EventList, []

def SimpleStaticEvents( boxes, DownsampleFactor ):
    
    EventList,EventBoxes = NMSSpace(boxes, DownsampleFactor)

    return  EventList, []
  
def NMSSpaceST(boxes, DownsampleFactor):
    
    
   
   if boxes is not None: 
    if len(boxes) == 0:
      return [], []
    else:
     boxes = np.array(boxes, dtype = float)
     assert boxes.shape[0] > 0
     if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float16)
  
    
     idxs = boxes[:,4].argsort()[::-1]
    
     pick = []
     Averageboxes = []
     while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
 
        threshold =  15
        distance = compute_dist_nt(boxes[i], boxes[idxs[1:]])
        remove_idxs = np.where(distance < threshold)[0] + 1
        
        weightX = boxes[i][4] * boxes[i][0]
        weightY = boxes[i][4] * boxes[i][1]
        weightXbox = boxes[i][4] * boxes[i][2]
        weightYbox = boxes[i][4] * boxes[i][3]
        weightscore = boxes[i][4]
        meanscore = boxes[i][4]
        label = boxes[i][6]
        weightTbox =  boxes[i][5]
        weightTraw = boxes[i][7]
        weightheight = boxes[i][4] * boxes[i][8]
        weightwidth = boxes[i][4] * boxes[i][9]
        if isinstance(remove_idxs, list):
          for idws in len(remove_idxs):
              weightX = weightX + boxes[idws][4] * boxes[idws][0]
              weightY = weightY + boxes[idws][4] * boxes[idws][1]
              weightXbox = weightXbox + boxes[idws][4] * boxes[idws][2]
              weightYbox = weightYbox + boxes[idws][4] * boxes[idws][3]
              weightscore = weightscore + boxes[idws][4]
              meanscore = max(meanscore,boxes[idws][4])
              weightTbox = min(weightTbox,  boxes[idws][5])
              weightTraw = min(weightTraw, boxes[idws][7])
              weightheight = weightheight + boxes[idws][4] * boxes[idws][8]
              weightwidth = weightwidth + boxes[idws][4] * boxes[idws][9]
        newbox = (weightX/weightscore,weightY/weightscore,weightXbox/weightscore,weightYbox/weightscore,meanscore,weightTbox,label,weightTraw,weightheight/weightscore,weightwidth/weightscore)
        
        Averageboxes.append(newbox) 
        idxs = np.delete(idxs, remove_idxs)
        idxs = np.delete(idxs, 0)
        
        
     centerlist = []    

     for i in range(len(Averageboxes)):
       center = ( ((Averageboxes[i][2]) *DownsampleFactor ) , ((Averageboxes[i][3]) * DownsampleFactor) )
       
       size =  math.sqrt(Averageboxes[i][8] * Averageboxes[i][8] + Averageboxes[i][9] * Averageboxes[i][9] ) * DownsampleFactor
       centerlist.append([center, Averageboxes[i][5],Averageboxes[i][6], Averageboxes[i][4], size] )
     Averageboxes = np.array(Averageboxes, dtype = float)
     return centerlist, []
 
def NMSSpace(boxes,DownsampleFactor):
    
   
   if boxes is not None: 
    if len(boxes) == 0:
      return [], []
    else:
     boxes = np.array(boxes, dtype = float)
     assert boxes.shape[0] > 0
     if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float16)
  
    
     idxs = boxes[:,4].argsort()[::-1]
    
     pick = []
     Averageboxes = []
     while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
 
        threshold =  50
        distance = compute_dist_nt(boxes[i], boxes[idxs[1:]])
        remove_idxs = np.where(distance < threshold)[0] + 1
        
        weightX = boxes[i][4] * boxes[i][0]
        weightY = boxes[i][4] * boxes[i][1]
        weightXbox = boxes[i][4] * boxes[i][2]
        weightYbox = boxes[i][4] * boxes[i][3]
        weightscore = boxes[i][4]
        meanscore = boxes[i][4]
        label = boxes[i][6]
        weightTbox =  boxes[i][5]
        weightTraw = boxes[i][7]
        weightheight = boxes[i][4] * boxes[i][8]
        weightwidth = boxes[i][4] * boxes[i][9]
        if isinstance(remove_idxs, list):
          for idws in len(remove_idxs):
              weightX = weightX + boxes[idws][4] * boxes[idws][0]
              weightY = weightY + boxes[idws][4] * boxes[idws][1]
              weightXbox = weightXbox + boxes[idws][4] * boxes[idws][2]
              weightYbox = weightYbox + boxes[idws][4] * boxes[idws][3]
              weightscore = weightscore + boxes[idws][4]
              meanscore = max(meanscore,boxes[idws][4])
              weightTbox = min(weightTbox,  boxes[idws][5])
              weightTraw = min(weightTraw, boxes[idws][7])
              weightheight = weightheight + boxes[idws][4] * boxes[idws][8]
              weightwidth = weightwidth + boxes[idws][4] * boxes[idws][9]
        newbox = (weightX/weightscore,weightY/weightscore,weightXbox/weightscore,weightYbox/weightscore,meanscore,weightTbox,label,weightTraw,weightheight/weightscore,weightwidth/weightscore)
        
        Averageboxes.append(newbox) 
        idxs = np.delete(idxs, remove_idxs)
        idxs = np.delete(idxs, 0)
        
        
     centerlist = []    

     for i in range(len(Averageboxes)):
       center = ( ((Averageboxes[i][2]) *DownsampleFactor ) , ((Averageboxes[i][3]) * DownsampleFactor) )
       
       size =  math.sqrt(Averageboxes[i][8] * Averageboxes[i][8] + Averageboxes[i][9] * Averageboxes[i][9] ) * DownsampleFactor
       centerlist.append([center, Averageboxes[i][5],Averageboxes[i][6], Averageboxes[i][4], size] )
     Averageboxes = np.array(Averageboxes, dtype = float)
     return centerlist, []
    
def Justbox(Averageboxes, DownsampleFactor):
    
    
   
     centerlist = []
     for i in range(len(Averageboxes)):
       center = ( ((Averageboxes[i][2]) *DownsampleFactor ) , ((Averageboxes[i][3]) * DownsampleFactor) )
       
       size =  math.sqrt(Averageboxes[i][8] * Averageboxes[i][8] + Averageboxes[i][9] * Averageboxes[i][9] ) * DownsampleFactor
       centerlist.append([center, Averageboxes[i][5],Averageboxes[i][6], Averageboxes[i][4], size] )
     Averageboxes = np.array(Averageboxes, dtype = float)
     return centerlist, []
 





def compute_dist_nt(box, boxes):
    # Calculate intersection areas
    
    Xtarget = boxes[:, 2]
    Ytarget = boxes[:, 3]
    
    Xsource = box[2]
    Ysource = box[3]
   
    # If seperated in time the distance is made lower to avoid multi counting of events
    
    distance = (Xtarget - Xsource) * (Xtarget - Xsource) + (Ytarget - Ysource) * (Ytarget - Ysource)
   
    
    return np.sqrt(distance)


def compute_dist_list(listA, listAs):
    # Calculate intersection areas
    
    Xtarget = listAs[:, 0]
    Ytarget = listAs[:, 1]
    
    Xsource = listA[0]
    Ysource = listA[1]
   
    # If seperated in time the distance is made lower to avoid multi counting of events
    
    distance = (Xtarget - Xsource) * (Xtarget - Xsource) + (Ytarget - Ysource) * (Ytarget - Ysource)
   
    
    return np.sqrt(distance)

    
def XYDistance(Xsource, Ysource, Xtarget, Ytarget):
    
    Xsource = Xsource
    Ysource = Ysource
    
    Xtarget = Xtarget
    Ytarget = Ytarget
    
    distance = (Xtarget - Xsource) * (Xtarget - Xsource) + (Ytarget - Ysource) * (Ytarget - Ysource)
    
    return np.sqrt(distance)