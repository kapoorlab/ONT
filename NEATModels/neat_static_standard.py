#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:13:01 2020

@author: Varun Kapoor
"""

from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from keras import callbacks
import os
from NEATModels import nets
from keras import backend as K
#from IPython.display import clear_output
from keras import optimizers
from pathlib import Path
import tensorflow as tf


class NEATStaticDetection(object):
    

    """
    Parameters
    ----------
    config : config is an object containing information about the network parameters, this object is also saved as a json file
    
    TrainDirectory : Specify the location of the directory containing the training data with movies and labels
    
    Categories_Name : List of class names and labels
    
    box_vector : Number of components other than the categories that the network has to output to perform localization
    
    model_dir : Directory location where trained model weights are to be read or written from
    
    model_name : The h5 file of CNN + LSTM + Dense Neural Network to be used for training
    
    model_weights : If re-training model_weights = model_dir + model_name else None as default
    
    show : If true the results of trainng will be displayed every epoch on a randomly chosen image from the validation set along with the loss and accuracy plot. THis option is set true
    if you are using jupyter notebooks but if executing this program from a py file this parameter is set to false.
    
    
    
    
    """
    
    
    def __init__(self, staticconfig, TrainDirectory, KeyCatagories, KeyCord, model_dir, model_name,  show = False ):

        self.TrainDirectory = TrainDirectory
        self.model_dir = model_dir
        self.model_name = model_name
        self.KeyCatagories = KeyCatagories
        self.box_vector = len(KeyCord)
        self.KeyCord = KeyCord
        self.model_weights = None
        self.show = show
       
        self.categories = len(KeyCatagories)
        self.depth = staticconfig.depth
        self.start_kernel = staticconfig.start_kernel
        self.mid_kernel = staticconfig.mid_kernel
        self.learning_rate = staticconfig.learning_rate
        self.epochs = staticconfig.epochs
        self.residual = staticconfig.residual
        self.startfilter = staticconfig.startfilter
        self.batch_size = staticconfig.batch_size
        self.multievent = staticconfig.multievent
        
        self.nboxes = staticconfig.nboxes
        self.gridX = staticconfig.gridX
        self.gridY = staticconfig.gridY
        self.lambdacord = staticconfig.lambdacord
        self.last_activation = None
        self.entropy = None
        self.X = None
        self.Y = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None

    def loadData(self):
        (X,Y), (X_val,Y_val) = helpers.load_full_training_data(self.TrainDirectory, self.categories, self.nboxes)

        self.X = X
        self.Y = Y
        self.X_val = X_val
        self.Y_val = Y_val
          

              
    def TrainModel(self):
        
        # input shape is  H W C
        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3])
        
        Path(self.model_dir).mkdir(exist_ok=True)
        
        
        
        if self.residual:
            model_keras = nets.resnet_v2
        else:
            model_keras = nets.seqnet_v2
        
        if self.multievent == True:
           self.last_activation = 'sigmoid'
           self.entropy = 'binary'
           
           
        if self.multievent == False:
           self.last_activation = 'softmax'              
           self.entropy = 'notbinary' 
           
        model_weights = self.model_dir + self.model_name
        if os.path.exists(model_weights):
        
            self.model_weights = model_weights
            print('loading weights')
        else:
           
            self.model_weights = None    
           
        
        self.Trainingmodel = model_keras(input_shape, self.categories, box_vector = self.box_vector , depth = self.depth, start_kernel = self.start_kernel, mid_kernel = self.mid_kernel, startfilter = self.startfilter, 
                                         gridX = self.gridX, gridY = self.gridY, nboxes = self.nboxes,  last_activation = self.last_activation, input_weights  =  self.model_weights)
        
            
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        
        self.Trainingmodel.compile(optimizer=sgd, loss = static_yolo_loss(self.categories, self.gridX, self.gridY, self.nboxes, self.box_vector, self.lambdacord, self.entropy, self.batch_size), metrics=['accuracy'])
        self.Trainingmodel.summary()
        print('Training Model:', model_keras)
        
        #Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotStaticHistory(self.Trainingmodel, self.X_val, self.Y_val, self.KeyCatagories, self.KeyCord, self.gridX, self.gridY, plot = self.show, nboxes = self.nboxes)
        
        
        #Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X,self.Y, batch_size = self.batch_size, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), shuffle = True, callbacks = [lrate,hrate,srate,prate])
     
        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name )
        
        self.Trainingmodel.save(self.model_dir + self.model_name )
        
    




def static_yolo_loss(categories, gridX, gridY, nboxes, box_vector, lambdacord, entropy, batch_size):
    
    def loss(y_true, y_pred):
        
        
        ANCHORS          = [0.11,0.11, 0.17,0.19, 0.26,0.29, 0.37,0.37, 0.62,0.55]
  
        OBJECT_SCALE = 1
        COORD_SCALE = lambdacord
        CLASS_SCALE = 1
        NO_OBJECT_SCALE = 1
        CLASS_WEIGHTS    = np.ones(categories, dtype='float32')
        WARM_UP_BATCHES  = 0

        mask_shape = tf.shape(y_true)[categories: categories + 4]
    
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(gridX), [gridY]), (1, gridY, gridX, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    
        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [batch_size, 1, 1, box_vector, 1])
        
        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
        
        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        
        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., categories: categories + 2]) + cell_grid
        
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., categories + 2: categories + 4]) * np.reshape(ANCHORS, [1,1,1,nboxes,2])
        
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., categories + 4])
        
        ### adjust class probabilities
        pred_box_class = y_pred[..., :categories]
        
        
        y_true_class = y_true[...,0:categories]
        y_pred_class = y_pred[...,0:categories]
        
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., categories: categories + 2] # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., categories + 2: categories + 4] # number of cells accross, horizontally and vertically
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        true_box_conf = iou_scores * y_true[..., categories + 4]
        
        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., :categories], -1)
        
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., categories : categories + 4], axis=-1) * COORD_SCALE
        
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        
        pred_boxes = K.reshape(y_pred[...,categories:], (-1, gridY * gridX, nboxes, box_vector))
        true_boxes = K.reshape(y_true[...,categories:], (-1, gridY * gridX, nboxes, box_vector))
        
        
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
    
        best_ious = tf.reduce_max(iou_scores, axis=-1)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., categories: categories + 4]) * NO_OBJECT_SCALE
        
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., categories: categories + 4] * OBJECT_SCALE
        
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., categories: categories + 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       
        
        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE/2.)
        seen = tf.assign_add(seen, 1.)
        
        
        
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                       true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,nboxes,2]) * no_boxes_mask, 
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       coord_mask])
        
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
        
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        if entropy == 'binary':
            loss_class = K.mean(K.binary_crossentropy(y_true_class, y_pred_class), axis=-1)
        else:
            loss_class = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis=-1)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        
        loss = loss_xy + loss_wh + loss_conf + loss_class
        print(loss_xy, loss_wh, loss_conf, loss_class)
        nb_true_box = tf.reduce_sum(y_true[..., 4])
        nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))
    
        """
        Debugging code
        """    
        current_recall = nb_pred_box/(nb_true_box + 1e-6)
        total_recall = tf.assign_add(total_recall, current_recall) 
    
        loss = tf.print(loss, [tf.zeros((1))], 'Dummy Line', summarize=1000)
        loss = tf.print(loss, [loss_xy],'Loss XY', summarize=1000)
        loss = tf.print(loss, [loss_wh],'Loss WH', summarize=1000)
        loss = tf.print(loss, [loss_conf],'Loss Conf', summarize=1000)
        loss = tf.print(loss, [loss_class],'Loss Class', summarize=1000)
        loss = tf.print(loss, [loss],'Total Loss', summarize=1000)
        loss = tf.print(loss, [current_recall],'Current Recall', summarize=1000)
        loss = tf.print(loss, [total_recall/seen], 'Average Recall', summarize=1000)
        
    return loss
    
    

def simple_static_yolo_loss(categories, gridX, gridY, nboxes, box_vector, lambdacord, entropy):
    
    def loss(y_true, y_pred):
        
       
        grid = np.array([ [[float(x),float(y)]]*nboxes   for y in range(gridY) for x in range(gridX)])
        
        y_true_class = y_true[...,0:categories]
        y_pred_class = y_pred[...,0:categories]
        
        
        pred_boxes = K.reshape(y_pred[...,categories:], (-1, gridY * gridX, nboxes, box_vector))
        true_boxes = K.reshape(y_true[...,categories:], (-1, gridY * gridX, nboxes, box_vector))
        
        y_pred_xy = pred_boxes[...,0:2] +  K.variable(grid)
        y_true_xy = true_boxes[...,0:2]
        
        y_pred_hw = pred_boxes[...,2:4]
        y_true_hw = true_boxes[...,2:4]
        
        
        
        if entropy == 'notbinary':
            class_loss = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis=-1)
        if entropy == 'binary':
            class_loss = K.mean(K.binary_crossentropy(y_true_class, y_pred_class), axis=-1)
        else:
            class_loss = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis=-1)
        xy_loss = K.sum(K.sum(K.square(y_true_xy - y_pred_xy), axis = -1) , axis = -1)
        hw_loss = K.sum(K.sum(K.square(K.sqrt(y_true_hw) - K.sqrt(y_pred_hw)), axis=-1), axis=-1)
      
        #IOU computation for increasing localization accuracy
       
        combinedloss =  class_loss + lambdacord * ( xy_loss + hw_loss )
        
        return combinedloss
    
    
    return loss
    
