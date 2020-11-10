from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from keras import callbacks
import os
from NEATModels import nets
from keras import backend as K
#from IPython.display import clear_output
from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
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


class NEATDetection(object):
    

    """
    Parameters
    ----------
    
    NpzDirectory : Specify the location of npz file containing the training data with movies and labels
    
    TrainModelName : Specify the name of the npz file containing training data and labels
    
    ValidationModelName :  Specify the name of the npz file containing validation data and labels
    
    categories : Number of action classes
    
    Categories_Name : List of class names and labels
    
    model_dir : Directory location where trained model weights are to be read or written from
    
    model_name : The h5 file of CNN + LSTM + Dense Neural Network to be used for training
    
    model_keras : The model as it appears as a Keras function
    
    model_weights : If re-training model_weights = model_dir + model_name else None as default
    
    lstm_hidden_units : Number of hidden uniots for LSTm layer, 64 by default
    
    epochs :  Number of training epochs, 55 by default
    
    batch_size : batch_size to be used for training, 20 by default
    
    
    
    """
    
    
    def __init__(self, config, NpzDirectory, TrainModelName, ValidationModelName, Categories_Name, box_vector, model_dir, model_name, model_weights = None,  show = False ):

        self.NpzDirectory = NpzDirectory
        self.TrainModelName = TrainModelName
        self.ValidationModelName = ValidationModelName
        self.model_dir = model_dir
        self.model_name = model_name
        self.Categories_Name = Categories_Name
        self.model_weights = model_weights
        self.show = show
        self.box_vector = box_vector
        self.categories = len(Categories_Name)
        self.depth = config.depth
        self.lstm_hidden_unit = config.lstm
        self.start_kernel = config.start_kernel
        self.mid_kernel = config.mid_kernel
        self.lstm_kernel = config.lstm_kernel
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs
        self.residual = config.residual
        self.simple = config.simple
        self.catsimple = config.catsimple
        self.startfilter = config.startfilter
        self.batch_size = config.batch_size
        
        self.anchors = config.anchors
        self.gridX = config.gridX
        self.gridY = config.gridY
        self.lambdacord = config.lambdacord
        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None
        print(self.startfilter)
    def loadData(self):
        
        (X,Y),  axes = helpers.load_full_training_data(self.NpzDirectory, self.TrainModelName, verbose= True)

        (X_val,Y_val), axes = helpers.load_full_training_data(self.NpzDirectory, self.ValidationModelName,  verbose= True)
        
        
        self.Xoriginal = X
        self.Xoriginal_val = X_val
        

                     

        self.X = X
        self.Y = Y[:,:,0]
        self.X_val = X_val
        self.Y_val = Y_val[:,:,0]
        self.axes = axes
        self.Y = self.Y.reshape( (self.Y.shape[0],1,1,self.Y.shape[1]))
        self.Y_val = self.Y_val.reshape( (self.Y_val.shape[0],1,1,self.Y_val.shape[1]))
          

              
    def TrainModel(self):
        
        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3], self.X.shape[4])
        
        
        Path(self.model_dir).mkdir(exist_ok=True)
        
        if self.residual == True and self.simple == False:
            model_keras = nets.ORNET
        if self.residual == False and self.simple == False: 
            model_keras = nets.OSNET
        if self.residual == True and self.simple == True:
            model_keras = nets.SimpleORNET
        if self.residual == False and self.simple == True:
            model_keras = nets.SimpleOSNET
        if self.residual == False and self.catsimple == True:
            model_keras = nets.CatSimpleOSNET
        if self.residual == True and self.catsimple == True:
            model_keras = nets.CatSimpleORNET
            
         
        self.Trainingmodel = model_keras(input_shape, self.categories,  unit = self.lstm_hidden_unit , box_vector = self.box_vector, gridX = self.gridX, gridY = self.gridY, depth = self.depth, start_kernel = self.start_kernel, mid_kernel = self.mid_kernel, startfilter = self.startfilter,  input_weights  =  self.model_weights)
        
            
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        
        
        
        if self.simple == False:
          self.Trainingmodel.compile(optimizer=sgd, loss = time_yolo_loss(self.categories, self.gridX, self.gridY, self.anchors, self.box_vector, self.lambdacord), metrics=['accuracy'])
        
        self.Trainingmodel.summary()
        print('Training Model:', model_keras)
        
        #Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotHistory(self.Trainingmodel, self.X_val, self.Y_val, self.Categories_Name, plot = self.show, simple = self.simple, catsimple = self.catsimple)
        
        
        #Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X,self.Y,batch_size = self.batch_size, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), shuffle = True, callbacks = [lrate,hrate,srate,prate])

     
        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name )
        
        self.Trainingmodel.save(self.model_dir + self.model_name )
        
        
    def plot_prediction(self, idx):
        
        helpers.Printpredict(idx, self.Trainingmodel, self.X_val, self.Y_val, self.Categories_Name)

   
def time_yolo_loss(categories, gridX, gridY, anchors, box_vector, lambdacord):
    
    def loss(y_true, y_pred):
        
       
        grid = np.array([ [[float(x),float(y), float(t)]]   for y in range(gridY) for x in range(gridX) for t in range(1)])
        
        y_true_class = y_true[...,0:categories]
        y_pred_class = y_pred[...,0:categories]
        
        
        pred_boxes = K.reshape(y_pred[...,categories:], (-1, gridY * gridX, anchors, box_vector))
        true_boxes = K.reshape(y_true[...,categories:], (-1, gridY * gridX, anchors, box_vector))
        
        y_pred_xyt = pred_boxes[...,0:3] +  (grid)
        y_true_xyt = true_boxes[...,0:3]
        
        y_pred_hw = pred_boxes[...,3:5]
        y_true_hw = true_boxes[...,3:5]
        
        y_pred_conf = pred_boxes[...,5]
        y_true_conf = true_boxes[...,5]
        
        
        class_loss = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis=-1)
        xy_loss = K.sum(K.sum(K.square(y_true_xyt - y_pred_xyt), axis = -1) * y_true_conf, axis = -1)
        hw_loss = K.sum(K.sum(K.square(K.sqrt(y_true_hw) - K.sqrt(y_pred_hw)), axis=-1)*y_true_conf, axis=-1)
      
        #IOU computation for increasing localization accuracy
       
        intersect_wh = K.maximum(K.zeros_like(y_pred_hw), (y_pred_hw + y_true_hw)/2 - K.square(y_pred_xyt - y_true_xyt) )
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
        true_area = y_true_hw[...,0] * y_true_hw[...,1]
        pred_area = y_pred_hw[...,0] * y_pred_hw[...,1]
        union_area = pred_area + true_area - intersect_area
        iou = intersect_area / union_area
        conf_loss = K.sum(K.square(y_true_conf*iou - y_pred_conf), axis=-1)

        combinedloss =  class_loss + lambdacord * xy_loss + hw_loss + conf_loss
        
        return combinedloss
    
    
    return loss
    


        
