#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:25:10 2020

@author: Varun Kapoor
"""
import argparse
import numpy as np

class NeatConfig(argparse.Namespace):
    
    def __init__(self, residual = True, gridX = 1, gridY = 1, nboxes = 1, lambdacord = 1, depth = 29, start_kernel = 3, mid_kernel = 3, lstm_kernel = 3, 
                 startfilter = 48, lstm = 16, epochs =100, sizeX = 256, sizeY = 256, sizeTminus = 5, sizeTplus = 3, categories = 6, box_vector = 7,
                 learning_rate = 1.0E-4, batch_size = 10, ModelName = 'NEATModel', multievent = True, Mode = 'Detection',TimeDistributedConv = True, ThreeDConv = True,  **kwargs):
        
        
           
           self.residual = residual
           self.depth = depth
           self.start_kernel = start_kernel
           self.mid_kernel = mid_kernel
           self.lstm_kernel = lstm_kernel
           self.startfilter = startfilter
           self.lstm = lstm
           self.epochs = epochs
           self.learning_rate = learning_rate
           self.batch_size = batch_size
           self.categories = categories
           self.box_vector = box_vector
           self.ModelName = ModelName
           self.lambdacord = lambdacord
           self.gridX = gridX
           self.gridY = gridY
           self.nboxes = nboxes
           self.multievent = multievent
           self.sizeX = sizeX
           self.sizeY = sizeY
           self.sizeTminus = sizeTminus
           self.sizeTplus = sizeTplus
           self.Mode = Mode
           self.TimeDistributedConv = TimeDistributedConv
           self.ThreeDConv = ThreeDConv
           self.is_valid()
    

    def to_json(self):

         config = {
                 'ModelName' : self.ModelName,
                 'residual' : self.residual,
                 'depth' : self.depth,
                 'start_kernel' : self.start_kernel,
                 'mid_kernel' : self.mid_kernel,
                 'lstm_kernel' : self.lstm_kernel,
                 'startfilter' : self.startfilter,
                 'lstm' : self.lstm,
                 'sizeX' : self.sizeX,
                 'sizeY' : self.sizeY,
                 'sizeTminus' : self.sizeTminus,
                 'sizeTplus' : self.sizeTplus,
                 'epochs' : self.epochs,
                 'learning_rate' : self.learning_rate,
                 'nboxes' : self.nboxes,
                 'gridX' : self.gridX,
                 'gridY' : self.gridY,
                 'lambdacord': self.lambdacord,
                 'batch_size' : self.batch_size,
                 'multievent' : self.multievent,
                 'Mode' : self.Mode,
                 'categories': self.categories,
                 'box_vector': self.box_vector,
                 'TimeDistributedConv' : self.TimeDistributedConv,
                 'ThreeDConv' : self.ThreeDConv
                 }
         return config
         
         
        
          
    def is_valid(self, return_invalid=False):
            """Check if configuration is valid.
            Returns
            -------
            bool
            Flag that indicates whether the current configuration values are valid.
            """
            def _is_int(v,low=None,high=None):
              return (
                isinstance(v,int) and
                (True if low is None else low <= v) and
                (True if high is None else v <= high)
              )

            ok = {}
            ok['residual'] = isinstance(self.residual,bool)
            ok['TimeDistributedConv'] = isinstance(self.TimeDistributedConv, bool)
            ok['ThreeDConv'] = isinstance(self.ThreeDConv, bool)
            ok['depth']         = _is_int(self.depth,1)
            ok['start_kernel']       = _is_int(self.start_kernel,1)
            ok['mid_kernel']         = _is_int(self.mid_kernel,1)
            ok['lstm_kernel']        = _is_int(self.lstm_kernel, 1)   
            ok['startfilter']        = _is_int(self.startfilter, 1)
            ok['lstm']         = _is_int(self.lstm,1)
            ok['epochs']        = _is_int(self.epochs, 1)
            ok['nboxes']       = _is_int(self.nboxes, 1)
            ok['gridX'] = _is_int(self.gridX, 1)
            ok['gridY'] = _is_int(self.gridY, 1)
            ok['lambdacord'] = _is_int(self.lambdacord, 1)
            ok['sizeX'] = _is_int(self.sizeX, 1)
            ok['sizeY'] = _is_int(self.sizeY, 1)
            ok['sizeTminus'] = _is_int(self.sizeTminus, 1)
            ok['sizeTplus'] = _is_int(self.sizeTplus, 1)
            ok['learning_rate'] = np.isscalar(self.learning_rate) and self.learning_rate > 0
            ok['multievent'] = isinstance(self.multievent,bool)
            ok['Mode'] = self.Mode in ('Detection', 'Prediction')
            ok['categories'] =  _is_int(self.categories, 1)
            ok['box_vector'] = _is_int(self.box_vector, 1)
    
            if return_invalid:
                return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
            else:
                return all(ok.values())
                   

           
           
           

        
