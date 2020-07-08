#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:25:10 2020

@author: aimachine
"""
import argparse
import numpy as np
class NeatConfig(argparse.Namespace):
    
    def __init__(self, allow_new_parameters = False, **kwargs):
        
        
        self.update_parameters(allow_new_parameters, **kwargs)
        
        
        
    def is_valid(self, return_invalid=False):
        return (True, tuple()) if return_invalid else True


    def update_parameters(self, allow_new=False, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])
            
            
            
class Config(NeatConfig):
        
        
        """ Default configuration of action classification model.
        
            Parameters
            ----------
            
            config = Config(startfilter = 48, start_kernel = 7, mid_kernel = 3, depth = 38 )
            
            
           Attributes
           ----------
           
           depth = depth of the network
           start_kernel = starting kernel size for the CNN and for the LSTM layer at the end
           mid_kernel = kernel size for the mid layers of the CNN
           batch_size = batch size for training
           lstm = number of hidden units for lstm
           categories = training categories
           box_vector = number of other yolo vectors used
           epochs = number of training epochs
           learning_rate = learning rate

        """
        
        def __init__(self, allow_new_parameters = False, **kwargs):
           
           super(Config, self).__init__()
           
           self.residual = True
           self.depth = 29
           self.start_kernel = 3
           self.mid_kernel = 3
           self.startfilter = 48
           self.lstm = 16
           self.epochs = 100
           self.learning_rate = 1.0E-4
           self.batch_size = 10
           
           self.update_parameters(allow_new_parameters, **kwargs)
           
           
           
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
            ok['depth']         = _is_int(self.depth,1)
            ok['start_kernel']       = _is_int(self.start_kernel,1)
            ok['mid_kernel']         = _is_int(self.mid_kernel,1)
            ok['startfilter']        = _is_int(self.startfilter, 1)
            ok['categories']         = _is_int(self.categories,1)
            ok['box_vector']        = _is_int(self.box_vector, 1)
            ok['lstm']        = _is_int(self.lstm, 1)
            ok['epochs']        = _is_int(self.epochs, 1)
            ok['learning_rate'] = np.isscalar(self.train_learning_rate) and self.train_learning_rate > 0
    
            if return_invalid:
                return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
            else:
                return all(ok.values())
               
           
           
           

        