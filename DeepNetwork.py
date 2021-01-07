# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:01:33 2021

@author: batho
"""

import numpy as np

class Network:
    
    def InitializeNetwork(self, layerDimensions):
        parameters={}
        if (len(layerDimensions)==0 or 
            layerDimensions[0]<=0 or type(layerDimensions) !=list):
            raise Exception("Can't initialize network with zero parameters")
        for layer in range(1,len(layerDimensions)):
            parameters["Weights"+str(layer)]=np.random.randn(
                layerDimensions[layer],layerDimensions[layer-1]) * 0.01
            parameters["bias"+str(layer)]=np.zeros((layerDimensions[layer], 1))
        return parameters
