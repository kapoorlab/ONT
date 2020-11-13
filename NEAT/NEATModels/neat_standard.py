from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from keras import callbacks
import os
from NEATModels import nets
from keras import backend as K
from keras import optimizers
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
    @author: Varun Kapoor
    
    Parameters
    ----------
    
    config : config is an object containing information about the network parameters, this object is also saved as a json file
    
    TrainDirectory : Specify the location of the directory containing the training data with movies and labels
    
    Categories_Name : List of class names and labels
    
    box_vector : Number of components other than the categories that the network has to output to perform localization
    
    model_dir : Directory location where trained model weights are to be read or written from
    
    model_name : The h5 file of CNN + LSTM + Dense Neural Network to be used for training
    
    model_weights : If re-training model_weights = model_dir + model_name else None as default
    
    show : If true the results of trainng will be displayed every epoch on a randomly chosen movie from the validation set along with the loss and accuracy plot. THis option is set true
    if you are using jupyter notebooks but if executing this program from a py file this parameter is set to false.
    
    
    
    """
    
    
    def __init__(self, config, TrainDirectory, Categories_Name, box_vector, model_dir, model_name, model_weights = None,  show = False ):

        self.TrainDirectory = TrainDirectory
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
        self.multievent = config.multievent
        self.startfilter = config.startfilter
        self.batch_size = config.batch_size
        self.anchors = config.anchors
        self.gridX = config.gridX
        self.gridY = config.gridY
        self.lambdacord = config.lambdacord
        self.last_activation = None
        self.entropy = None
        self.X = None
        self.Y = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
    def loadData(self):
        
        (X,Y), (X_val,Y_val) = helpers.load_full_training_data(self.TrainDirectory, self.categories, self.anchors)

        self.X = X
        self.Y = Y
        self.X_val = X_val
        self.Y_val = Y_val
              
    def TrainModel(self):
        
        # input shape is T H W C
        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3], self.X.shape[4])
        
        
        Path(self.model_dir).mkdir(exist_ok=True)
        
        if self.residual == True:
            model_keras = nets.ORNET
        if self.residual == False: 
            model_keras = nets.OSNET
            
        if self.multievent == True:
           self.last_activation = 'sigmoid'
           self.entropy = 'binary'
           
           
        if self.multievent == False:
           self.last_activation = 'softmax'              
           self.entropy = 'notbinary' 
         
        self.Trainingmodel = model_keras(input_shape, self.categories,  unit = self.lstm_hidden_unit , box_vector = self.box_vector, gridX = self.gridX, gridY = self.gridY, anchors = self.anchors, depth = self.depth, start_kernel = self.start_kernel,
                                         mid_kernel = self.mid_kernel, lstm_kernel = self.lstm_kernel, startfilter = self.startfilter,  
                                         input_weights  =  self.model_weights, last_activation = self.last_activation)
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        self.Trainingmodel.compile(optimizer = sgd, loss = time_yolo_loss(self.categories, self.gridX, self.gridY, self.anchors, self.box_vector, self.lambdacord, self.entropy), metrics=['accuracy'])
        self.Trainingmodel.summary()
        print('Training Model:', model_keras)
        
        #Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotHistory(self.Trainingmodel, self.X_val, self.Y_val, self.Categories_Name, self.gridX, self.gridY, plot = self.show)
        
        
        #Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X,self.Y,batch_size = self.batch_size, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), shuffle = True, callbacks = [lrate,hrate,srate,prate])

     
        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name)
        
        self.Trainingmodel.save(self.model_dir + self.model_name)
        
        
   
def time_yolo_loss(categories, gridX, gridY, anchors, box_vector, lambdacord, entropy):
    
    def loss(y_true, y_pred):
        
       
        grid = np.array([ [[float(x),float(y), float(t)]]*anchors   for y in range(gridY) for x in range(gridX) for t in range(1)])
        
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
        
        y_pred_angle = pred_boxes[...,6]
        y_true_angle = pred_boxes[...,6]
        
        if entropy == 'notbinary':
            class_loss = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis=-1)
        if entropy == 'binary':
            class_loss = K.mean(K.binary_crossentropy(y_true_class, y_pred_class), axis=-1)
        else:
            class_loss = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis=-1)
            
        xy_loss = K.sum(K.sum(K.square(y_true_xyt - y_pred_xyt), axis = -1) * y_true_conf, axis = -1)
        hw_loss = K.sum(K.sum(K.square(K.sqrt(y_true_hw) - K.sqrt(y_pred_hw)), axis=-1)*y_true_conf, axis=-1)
        angle_loss = K.sum(K.square(y_true_angle - y_pred_angle), axis=-1)
        
        #IOU computation for increasing localization accuracy
       
        intersect_wh = K.maximum(K.zeros_like(y_pred_hw), (y_pred_hw + y_true_hw)/2 - K.square(y_pred_xyt - y_true_xyt))
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
        true_area = y_true_hw[...,0] * y_true_hw[...,1]
        pred_area = y_pred_hw[...,0] * y_pred_hw[...,1]
        union_area = pred_area + true_area - intersect_area
        iou = intersect_area / union_area
        conf_loss = K.sum(K.square(y_true_conf*iou - y_pred_conf), axis=-1)

        combinedloss =  class_loss + lambdacord * (xy_loss + hw_loss) + conf_loss + angle_loss
        
        return combinedloss
    
    
    return loss
    


        
