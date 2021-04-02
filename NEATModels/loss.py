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
lambdaclass = 1
grid_h = 1
grid_w = 1

def get_cell_grid(grid_h, grid_w, batch_size, boxes):
    
    cell_grid = np.array([ [[float(x),float(y)]]*boxes   for y in range(grid_h) for x in range(grid_w)])
    
    return cell_grid
    
def adjust_scale_prediction(y_pred, cell_grid, anchors,grid_h, grid_w, box_vector):
    
    boxes = int(len(anchors)/2)
    pred_nboxes = K.reshape(y_pred[...,:box_vector], (-1, grid_h * grid_w, boxes, box_vector))
    
    pred_box_xy = (pred_nboxes[..., :2]) + cell_grid
    
    pred_box_wh = (pred_nboxes[..., 2:4]) 
    
    pred_box_conf = (y_pred[..., 4])
    
    pred_box_class = y_pred[..., 5:]
    
    return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class

def extract_ground_truth(y_true,grid_h, grid_w, box_vector, nboxes):

    true_nboxes = K.reshape(y_true[...,:box_vector], (-1, grid_h * grid_w, nboxes, box_vector))
    
    true_box_xy    = true_nboxes[..., 0:2]  
    true_box_wh    = true_nboxes[..., 2:4] 
    true_box_conf  = y_true[...,4]
    true_box_class = y_true[..., 5:]

    return true_box_xy, true_box_wh, true_box_conf, true_box_class
    
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

def calc_loss_xywh(true_box_conf, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh):

    
    coord_mask  = K.expand_dims(true_box_conf, axis=-1) * lambdacoord
    loss_xy      = K.sum(K.square(true_box_xy-pred_box_xy) * coord_mask)
    loss_wh      = K.sum(K.square(true_box_wh-pred_box_wh) * coord_mask) 
    loss_xywh = (loss_xy + loss_wh) / 2
    return loss_xywh, coord_mask

def calc_loss_class(true_box_conf, true_box_class, pred_box_class, entropy):

    
    class_mask   = true_box_conf  * lambdaclass
    
    if entropy == 'binary':
        loss_class = K.sum(K.binary_crossentropy(true_box_class, pred_box_class), axis=-1)
    if entropy == 'notbinary':
         
        loss_class   = K.sum(K.categorical_crossentropy(true_box_class, pred_box_class), axis=-1)
    loss_class   = loss_class * class_mask 

    return loss_class



def static_yolo_loss(categories, grid_h, grid_w, anchors, box_vector, entropy, batch_size):
    
    def loss(y_true, y_pred):    

            boxes = int(len(anchors)/2)
            
            # Get the cell grid
            cell_grid = get_cell_grid(grid_h, grid_w, batch_size, boxes)  
            
            #Extract the ground truth 
            true_box_xy, true_box_wh, true_box_conf, true_box_class = extract_ground_truth(y_true,grid_h, grid_w, box_vector, boxes)
            
            #Scale the prediction variables
            pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_scale_prediction(y_pred, cell_grid, anchors,grid_h, grid_w, box_vector)
        
            # xyhw loss
            loss_xywh, coord_mask = calc_loss_xywh(true_box_conf, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh)
            
            # class loss
            loss_class = calc_loss_class(true_box_conf, true_box_class, pred_box_class, entropy)
            
            # compute iou
            true_box_conf_iou = compute_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)
            
            # best ious
            best_ious = compute_best_ious(pred_box_xy, pred_box_wh, true_box_xy, true_box_wh)
            
            #conf loss
            loss_conf = compute_conf_loss(best_ious, true_box_conf, true_box_conf_iou, pred_box_conf)
            
           

            # Adding it all up   
            combinedloss = (loss_xywh + loss_conf + loss_class)
        

     
            return combinedloss 
        
    return loss    
        
        