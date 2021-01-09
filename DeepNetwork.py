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
    
    def sigmoidDifferential(self,activatedGradient,linearResult ):
        sigmoid= 1/(1+np.exp(-linearResult))
        linearGradient=activatedGradient*sigmoid*(1-sigmoid)
        return linearGradient
    
    def relu(self, linearResult):
        activatedResult = np.maximum(0,linearResult)
        return activatedResult
    
    def reluDifferential(self,activatedGradient, linearResult):
        linearGradient=np.array(activatedGradient, copy=True)
        linearGradient[linearResult <= 0] = 0
        return linearGradient
    
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
    
    def ComputeCost(self, finalActivation, targetSet):
        totalEntries=targetSet.shape[1]
        logProduct=np.dot(targetSet, np.log(finalActivation).T) + np.dot((1-targetSet),
                                                                         np.log(1-finalActivation).T)
        cost=-(1/totalEntries)*np.sum(logProduct)
        cost = np.squeeze(cost)
        return cost
        
    def LinearBackward(self, linearGradient,linearCache):
        previousActivation, weights, bias= linearCache
        totalEntries=previousActivation.shape[1]
        weightsGradient=(1/totalEntries)*np.dot(linearGradient,previousActivation.T)
        biasGradient=(1/totalEntries)*np.sum(linearGradient,axis=1,keepdims=True)
        previousActivationGradient=np.dot(weights.T,linearGradient)
        return previousActivationGradient, weightsGradient, biasGradient
    
    def LinearBackwardActivation(self,activatedGradient,cache,activation):
        linearCache,linearResult=cache
        if activation=="relu":
            linearGradient=self.reluDifferential(activatedGradient,linearResult)
        elif activation=="sigmoid":
            linearGradient=self.sigmoidDifferential(activatedGradient,linearResult)
        previousActivationGradient,weightsGradient,biasGradient=\
            self.LinearBackward(linearGradient,linearCache)
        return previousActivationGradient,weightsGradient,biasGradient 
    
    def BackPropagate(self, lastLayerActivation,targetSet,caches):
       gradients = {}
       totalLayers= len(caches)
       totalSamples=lastLayerActivation.shape[1]
       targetSet.reshape(lastLayerActivation.shape)
       lastLayerActivatedGradient=- (1/totalSamples)*(np.divide(targetSet, lastLayerActivation) \
                                  - np.divide(1 - targetSet, 1 - lastLayerActivation))
       currentCache=caches[totalLayers-1]
       gradients["ActivatedGradient"+str(totalLayers-1)], gradients["WeightsGradient" + str(totalLayers)],\
           gradients["BiasGradient" + str(totalLayers)] = \
               self.LinearBackwardActivation(lastLayerActivatedGradient, currentCache, "sigmoid")
       for layer in reversed(range(totalLayers-1)):
           currentCache=caches[layer]
           previousActivatedGradient, weightsGradient,biasGradient= \
               self.LinearBackwardActivation(gradients["ActivatedGradient"]+str(layer+1),
                                             currentCache,"relu")
           gradients["ActivatedGradient" + str(layer)] = previousActivatedGradient
           gradients["WeightsGradient" + str(layer + 1)] = weightsGradient
           gradients["BiasGradient" + str(layer + 1)] = biasGradient
       return gradients
   
    def UpdateWeights(self, parameters,gradients,learningRate=0.1):
        totalLayer= len(parameters)//2
        for layer in range(totalLayer):
            parameters["Weights" + str(layer+1)] = parameters["Weights" + str(layer+1)]\
                -learningRate*gradients["WeightsGradient" + str(layer + 1)]
            parameters["bias" + str(layer+1)] = parameters["bias" + str(layer+1)]-\
                learningRate*gradients["BiasGradient" + str(layer + 1)]
        return parameters
        
    
    
        
            
        
    
