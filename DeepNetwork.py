# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:01:33 2021

@author: batho
"""

import numpy as np

class Network:
    
    def dataReshape(self,dataset):
        return dataset.reshape(dataset.shape[0],-1).T
    
    def sigmoid(self,linearResult):
        activatedResult= 1/(1+np.exp(-linearResult))
        return activatedResult
    
    def relu(self, linearResult):
        activatedResult = np.maximum(0,linearResult)
        return activatedResult
    
    def InitializeNetwork(self, layerDimensions):
        parameters={}
        if (len(layerDimensions)==0 or 
            0 in layerDimensions or type(layerDimensions) !=list):
            raise Exception("Can't initialize network with zero parameters")
        for layer in range(1,len(layerDimensions)):
            parameters["Weights"+str(layer)]=np.random.randn(
                layerDimensions[layer],layerDimensions[layer-1]) * 0.01
            parameters["bias"+str(layer)]=np.zeros((layerDimensions[layer], 1))
        return parameters
    
    def GetLinearResult(self,previousActivation, weights, bias ):
        linearResult=np.dot(weights,previousActivation)+bias
        cache=(previousActivation, weights, bias)
        return linearResult, cache
    
    def ActivateLinearResult(self, previousActivation, weight, bias, activation):
        linearResult, linearCache = self.GetLinearResult(previousActivation,
                                                          weight, bias)
        if activation == "sigmoid":
            activatedResult=self.sigmoid(linearResult)
        elif activation == "relu":
           activatedResult=self.relu(linearResult)
        cache=(linearCache,linearResult)
        return activatedResult,cache
    
    def ForwardPropagate(self,trainData,parameters):
        caches = []
        activatedResult = trainData
        totalLayers=len(parameters) // 2
        for layer in range(1,totalLayers):
            previousActivation=activatedResult
            activatedResult,cache=self.ActivateLinearResult(previousActivation,
                                                      parameters['Weights'+str(layer)],
                                                      parameters['bias'+str(layer)], 
                                                      "relu")
            caches.append(cache)
        lastLayerActivation,cache=self.ActivateLinearResult(activatedResult,
                                                            parameters['Weights'
                                                                       +str(totalLayers)],
                                                            parameters['bias'+str(totalLayers)],
                                                            "sigmoid")
        caches.append(cache)
        
        return lastLayerActivation,caches
    
        
            
        
    
