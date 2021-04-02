#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:05:34 2021

@author: vkapoor
"""

import tensorflow as tf
import numpy as np
import os, sys
from keras import backend as K

lambdaobject = 1
lambdanoobject = 1
lambdacoord = 1
lambdaclass = 5


def get_cell_grid(grid_h, grid_w, boxes):
    
    cell_grid =  np.array([ [[float(x),float(y)]]*boxes   for y in range(grid_h) for x in range(grid_w)])
    
    return cell_grid
    
def adjust_scale_prediction(y_pred, cell_grid, boxes,grid_h, grid_w, categories, box_vector):
    
        pred_boxes = K.reshape(y_pred[...,categories:], (-1,grid_w*grid_h,boxes,box_vector))

        pred_box_xy   = pred_boxes[...,0:2] +(cell_grid)
        # w and h predicted are 0 to 1 with 1 being image size
        pred_box_wh   = pred_boxes[...,2:4]
        # probability that there is something to predict here
        #pred_box_conf = pred_boxes[...,4]
   
        pred_box_class = y_pred[..., :categories]
    
        return pred_box_xy, pred_box_wh, pred_box_class

def extract_ground_truth(y_true,grid_h, grid_w, box_vector,boxes, categories):

    
    true_boxes = K.reshape(y_true[...,categories:], (-1,grid_w*grid_h,boxes,box_vector))
    true_box_xy    = true_boxes[...,0:2]
    true_box_wh    = true_boxes[...,2:4]
    #true_box_conf  = true_boxes[...,4]
    true_box_class = y_true[..., :categories]

    return true_box_xy, true_box_wh, true_box_class
    
def compute_iou(true_xy, true_wh, pred_xy, pred_wh):

       
    intersect_wh = K.maximum(K.zeros_like(pred_wh), (pred_wh + true_wh)/2 - K.square(pred_xy - true_xy))
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
    true_area = true_wh[...,0] * true_wh[...,1]
    pred_area = pred_wh[...,0] * pred_wh[...,1]
    union_area = pred_area + true_area - intersect_area
    iou_scores = tf.truediv(intersect_area , union_area)

    
    return iou_scores

def compute_boxconf_iou(true_box_conf, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):

    iou_scores        =  compute_iou(true_box_xy,true_box_wh,
                                            pred_box_xy,pred_box_wh)
    true_box_conf_IOU = iou_scores * true_box_conf
    
    return true_box_conf_IOU

def compute_best_ious(pred_box_xy, pred_box_wh, true_xy, true_wh):

    
    
    iou_scores  =  compute_iou(true_xy,
                                      true_wh,
                                      pred_box_xy,
                                      pred_box_wh)   

    best_ious = K.max(iou_scores, axis= -1)
    
    return best_ious

def compute_conf_loss(best_ious, true_box_conf, true_box_conf_iou,pred_box_conf):
    
    indicator_noobj = K.cast(best_ious < 0.6, np.float32) * (1 - true_box_conf) * lambdanoobject
    indicator_obj = true_box_conf * lambdaobject
    indicator_o = indicator_obj + indicator_noobj

    loss_obj = K.sum(K.square(true_box_conf_iou-pred_box_conf) * indicator_o)#, axis=[1,2,3])

    return loss_obj / 2

def calc_loss_xywh( true_box_xy, pred_box_xy, true_box_wh, pred_box_wh):

    
    coord_mask  =  lambdacoord
    loss_xy      = K.sum(K.square(true_box_xy-pred_box_xy) )
    loss_wh      = K.sum(K.square(true_box_wh-pred_box_wh) ) 
    loss_xywh = coord_mask * (loss_xy + loss_wh) / 2
    return loss_xywh, coord_mask

def calc_loss_class(true_box_class, pred_box_class, entropy):

    
    class_mask   =  lambdaclass
    
    if entropy == 'binary':
        loss_class = K.sum(K.binary_crossentropy(true_box_class, pred_box_class), axis=-1)
    if entropy == 'notbinary':
         
        loss_class   = K.sum(K.categorical_crossentropy(true_box_class, pred_box_class), axis=-1)
    loss_class   = loss_class * class_mask 

    return loss_class


def simple_yolo_loss(categories, grid_h, grid_w, boxes, box_vector, entropy, batch_size):
    
    def loss(y_true, y_pred):
        
        grid = np.array([ [[float(x),float(y)]]*boxes   for y in range(grid_h) for x in range(grid_w)])

        # first three values are classes : cat, rat, and none.
        # However yolo doesn't predict none as a class, none is everything else and is just not predicted
        # so I don't use it in the loss
        y_true_class = y_true[...,:categories]
        y_pred_class = y_pred[...,:categories] 

        # reshape array as a list of grid / grid cells / boxes / of 5 elements
        pred_boxes = K.reshape(y_pred[...,categories:], (-1,grid_w*grid_h,boxes,5))
        true_boxes = K.reshape(y_true[...,categories:], (-1,grid_w*grid_h,boxes,5))

        # sum coordinates of center of boxes with cell offsets.
        # as pred boxes are limited to 0 to 1 range, pred x,y + offset is limited to predicting elements inside a cell
        y_pred_xy   = pred_boxes[...,0:2] +(grid)
        # w and h predicted are 0 to 1 with 1 being image size
        y_pred_wh   = pred_boxes[...,2:4]
        # probability that there is something to predict here
        y_pred_conf = pred_boxes[...,4]

        # same as predicate except that we don't need to add an offset, coordinate are already between 0 and cell count
        y_true_xy   = true_boxes[...,0:2]
        # with and height
        y_true_wh   = true_boxes[...,2:4]
        # probability that there is something in that cell. 0 or 1 here as it's a certitude.
        y_true_conf = true_boxes[...,4]

        clss_loss  = K.sum(K.square(y_true_class - y_pred_class), axis=-1)
        xy_loss    = K.sum(K.sum(K.square(y_true_xy - y_pred_xy),axis=-1)*y_true_conf, axis=-1)
        wh_loss    = K.sum(K.sum(K.square(K.sqrt(y_true_wh) - K.sqrt(y_pred_wh)), axis=-1)*y_true_conf, axis=-1)

        # when we add the confidence the box prediction lower in quality but we gain the estimation of the quality of the box
        # however the training is a bit unstable

        # compute the intersection of all boxes at once (the IOU)
        intersect_wh = K.maximum(K.zeros_like(y_pred_wh), (y_pred_wh + y_true_wh)/2 - K.square(y_pred_xy - y_true_xy) )
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
        true_area = y_true_wh[...,0] * y_true_wh[...,1]
        pred_area = y_pred_wh[...,0] * y_pred_wh[...,1]
        union_area = pred_area + true_area - intersect_area
        iou = intersect_area / union_area

        conf_loss = K.sum(K.square(y_true_conf*iou - y_pred_conf), axis=-1)

        # final loss function
        combinedloss =  (xy_loss + wh_loss) + conf_loss + clss_loss
    
    
    
        
        
        return combinedloss 
        
    return loss


def static_yolo_loss(categories, grid_h, grid_w, boxes, box_vector, entropy, batch_size):
    
    def loss(y_true, y_pred):    

            
            # Get the cell grid
            cell_grid = get_cell_grid(grid_h, grid_w, boxes)  
            
            #Extract the ground truth 
            true_box_xy, true_box_wh,  true_box_class = extract_ground_truth(y_true,grid_h, grid_w, box_vector,boxes, categories)
            
            #Scale the prediction variables
            pred_box_xy, pred_box_wh,  pred_box_class = adjust_scale_prediction(y_pred, cell_grid, boxes,grid_h, grid_w, categories, box_vector)
        
            # xyhw loss
            loss_xywh, coord_mask = calc_loss_xywh(true_box_xy, pred_box_xy, true_box_wh, pred_box_wh)
            
            # class loss
            loss_class = calc_loss_class(true_box_class, pred_box_class, entropy)
            
            # compute iou
            #true_box_conf_iou = compute_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)
            
            # best ious
            #best_ious = compute_best_ious(pred_box_xy, pred_box_wh, true_box_xy, true_box_wh)
            
            #conf loss
            #loss_conf = compute_conf_loss(best_ious, true_box_conf, true_box_conf_iou, pred_box_conf)
            
           

            # Adding it all up   
            combinedloss = (loss_xywh + loss_class)
        

     
            return combinedloss 
        
    return loss    
        