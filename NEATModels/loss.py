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
lambdacoord = 2
lambdaclass = 1
grid_h = 1
grid_w = 1

def get_cell_grid(grid_h, grid_w, batch_size, boxes):
    
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [batch_size, 1, 1, boxes, 1])
    
    return cell_grid
    
def adjust_scale_prediction(y_pred, cell_grid, anchors):
    
    boxes = int(len(anchors)/2)
    
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(anchors,[1,1,1,boxes,2])
    
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    
    pred_box_class = y_pred[..., 5:]
    
    return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class

def extract_ground_truth(y_true):

    true_box_xy    = y_true[..., 0:2]  
    true_box_wh    = y_true[..., 2:4] 
    true_box_conf  = y_true[...,4]
    true_box_class = y_true[..., 5:]

    return true_box_xy, true_box_wh, true_box_conf, true_box_class
    
def compute_iou(true_xy, true_wh, pred_xy, pred_wh):

    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    
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
    
    return iou_scores

def compute_boxconf_iou(true_box_conf, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):

    iou_scores        =  compute_iou(true_box_xy,true_box_wh,
                                            pred_box_xy,pred_box_wh)
    true_box_conf_IOU = iou_scores * true_box_conf
    
    return true_box_conf_IOU

def compute_best_ious(pred_box_xy, pred_box_wh, true_xy, true_wh):

    
    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)
    
    iou_scores  =  compute_iou(true_xy,
                                      true_wh,
                                      pred_xy,
                                      pred_wh)   

    best_ious = tf.reduce_max(iou_scores, axis=4)
    
    return best_ious

def compute_conf_mask(best_ious, true_box_conf, true_box_conf_iou, lambdanoobject, lambdaobject):
    
    
    conf_mask = tf.to_float(best_ious < 0.6) * (1 - true_box_conf) * lambdanoobject
    conf_mask = conf_mask + true_box_conf_iou * lambdanoobject

    return conf_mask

def calc_loss_xywh(true_box_conf, lambdacoord, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh):

    
    coord_mask  = tf.expand_dims(true_box_conf, axis=-1) * lambdacoord 
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    loss_xy      = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh      = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_xywh = loss_xy + loss_wh
    return loss_xywh, coord_mask

def calc_loss_class(true_box_conf, lambdaclass, true_box_class, pred_box_class, entropy):

    
    class_mask   = true_box_conf  * lambdaclass
    
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    
    
    if entropy == 'binary':
        loss_class = K.mean(K.binary_crossentropy(true_box_class, pred_box_class), axis=-1)
    if entropy == 'notbinary':
         
        loss_class   = K.mean(K.categorical_crossentropy(true_box_class, pred_box_class), axis=-1)
    loss_class   = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    return loss_class


def compute_conf_loss(conf_mask, true_box_conf_iou, pred_box_conf):
    
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    loss_conf    = tf.reduce_sum(tf.square(true_box_conf_iou-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.

    return loss_conf

def static_yolo_loss(categories, grid_h, grid_w, anchors, box_vector, entropy, batch_size):
    
    def loss(y_true, y_pred):    

            boxes = int(len(anchors)/2)
            
            # Get the cell grid
            cell_grid = get_cell_grid(grid_h, grid_w, batch_size, boxes)  
            
            #Extract the ground truth 
            true_box_xy, true_box_wh, true_box_conf, true_box_class = extract_ground_truth(y_true)
            
            #Scale the prediction variables
            pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_scale_prediction(y_pred, cell_grid, anchors)
        
            # xyhw loss
            loss_xywh, coord_mask = calc_loss_xywh(true_box_conf, lambdacoord, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh)
            
            # class loss
            loss_class = calc_loss_class(true_box_conf, lambdaclass, true_box_class, pred_box_class, entropy)
            
            # compute iou
            true_box_conf_iou = compute_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)
            
            # best ious
            best_ious = compute_best_ious(pred_box_xy, pred_box_wh, true_box_xy, true_box_wh)
            
            # conf mask
            conf_mask = compute_conf_mask(best_ious, true_box_conf, true_box_conf_iou, lambdanoobject, lambdaobject)
            
            #conf loss
            loss_conf = compute_conf_loss(conf_mask, true_box_conf_iou, pred_box_conf)

            # Adding it all up   
            combinedloss = loss_xywh + loss_conf + loss_class
        

     
            return combinedloss 
        
    return loss    
        
        