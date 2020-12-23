#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:12:49 2020

@author: kapoorlab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:25:10 2020

@author: Varun Kapoor
"""
import argparse

class TrainConfig(argparse.Namespace):
    
    def __init__(self, ActionEventCSV, ActionEventLabel,  **kwargs):
        
        
           
           self.ActionEventCSV = ActionEventCSV
           self.ActionEventLabel = ActionEventLabel
           assert len(ActionEventCSV) == len(ActionEventLabel)
    

    def to_json(self):

         config = {}
         for i in range(0, len(self.ActionEventCSV)):
             
             config[self.ActionEventCSV[i]] = self.ActionEventLabel[i]
             
        
         return config
         
         
        
          
   

        
