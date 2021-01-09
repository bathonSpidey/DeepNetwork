# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:21:42 2021

@author: batho
"""
import DeepNetwork
import unittest
import numpy as np

class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.dataset=np.array([(0,0),(0,1),(1,0),(1,1)])
        self.testDataset=self.dataset
        self.targetSet=np.array([(1,0,0,1)])
        self.network=DeepNetwork.Network()
        self.trainData=self.network.dataReshape(self.dataset)
    
    def test_InitilaizeNetworkWithInvalidInput(self):
        with self.assertRaises(Exception):self.network.InitializeNetwork("1,2,4,3")
        with self.assertRaises(Exception):self.network.InitializeNetwork([])
        with self.assertRaises(Exception):self.network.InitializeNetwork([0,2,1])
        
    def test_GetParameters(self):
        parameters=self.network.InitializeNetwork([2,2,1])
        self.assertEqual(len(parameters.keys()),4)
        self.assertEqual(parameters["Weights1"].shape,(2,2))
        self.assertEqual(parameters["Weights2"].shape,(1,2))
        self.assertEqual(parameters["bias1"].shape,(2,1))
        self.assertEqual(parameters["bias2"].shape,(1,1))
        
    def test_GetParamtersOfDeeperNetwork(self):
        parameters=self.network.InitializeNetwork([2,8,2,6,1])
        self.assertEqual(len(parameters.keys()),8)
        self.assertEqual(parameters["Weights1"].shape,(8,2))
        self.assertEqual(parameters["Weights2"].shape,(2,8))
        self.assertEqual(parameters["Weights3"].shape,(6,2))
        self.assertEqual(parameters["Weights4"].shape,(1,6))
        self.assertEqual(parameters["bias1"].shape,(8,1))
        self.assertEqual(parameters["bias2"].shape,(2,1))
    
    def test_forwardPropagateForFirstLayerWithOneNode(self):
        parameters=self.network.InitializeNetwork([2,1,1])
        linearResult,_=self.network.GetLinearResult(self.trainData,
                                                         parameters["Weights1"],
                                                         parameters["bias1"])
        self.assertEqual(linearResult.shape,(1,4))
    
    def test_activateFirstLayerWithSigmoid(self):
        activatedResult,_,_=self.initializeActivationFunction("sigmoid",[2,1,1])
        self.assertEqual(activatedResult.shape,(1,4))
    
    def initializeActivationFunction(self, activation, layerList):
        parameters=self.network.InitializeNetwork(layerList)
        activatedResult,cache=self.network.ActivateLinearResult(
            self.trainData,parameters["Weights1"],parameters["bias1"], activation)
        return activatedResult,cache,parameters
        
    def test_activateFirstLayerWithRelu(self):
        activatedResult,cache,_=self.initializeActivationFunction("relu",[2,1,1])
        self.assertEqual(activatedResult.shape,(1,4))
        return cache
    
    def test_activatedResultForShallowNetwork(self):
        _,_,parameters=self.initializeActivationFunction("relu",[2,1,1])
        finalLayerActivation,caches=self.network.ForwardPropagate(self.trainData,
                                                                  parameters)
        self.assertEqual(finalLayerActivation.shape,(1,4))
    
    def test_activatedResultForDeepNetwork(self):
        _,_,parameters=self.initializeActivationFunction("relu",[2,8,4,2,1])
        finalLayerActivation,caches=self.network.ForwardPropagate(self.trainData,
                                                                  parameters)
        self.assertEqual(parameters["Weights"+str(len(parameters) // 2)].shape,(1,2))
        self.assertEqual(finalLayerActivation.shape,(1,4))
        return finalLayerActivation
    
    def test_costFinalLayer(self):
        activation=self.test_activatedResultForDeepNetwork()
        cost=self.network.ComputeCost(activation,self.targetSet)
        self.assertEqual(cost.shape,())
    
    def test_linearBackwardActivation(self):
        parameters=self.network.InitializeNetwork([2,1])
        lastLayerActivation, caches=self.network.ForwardPropagate(self.trainData,
                                                                  parameters)
        totalSamples=lastLayerActivation.shape[1]
        lastGradient=-(1/totalSamples)*(np.divide(self.targetSet, lastLayerActivation) \
                                  - np.divide(1 - self.targetSet, 1 - lastLayerActivation))
        lastCache=caches[len(caches)-1]
        linearCache,linearResult=lastCache
        previousActivation, weights, bias= linearCache
        previousActivationGradient,weightsGradient,biasGradient=\
            self.network.LinearBackwardActivation(lastGradient, lastCache,"sigmoid")
        self.assertEqual(previousActivationGradient.shape,previousActivation.shape)
        self.assertEqual(weightsGradient.shape,weights.shape)
        self.assertEqual(biasGradient.shape,bias.shape)
        
    def test_backPropagate(self):
        parameters=self.network.InitializeNetwork([2,1])
        lastLayerActivation, caches=self.network.ForwardPropagate(self.trainData,
                                                                  parameters)
        gradients=self.network.BackPropagate(lastLayerActivation,
                                             self.targetSet,caches)
        self.assertEqual(len(gradients),3)
        self.assertEqual(gradients["ActivatedGradient0"].shape,(2,4))
        self.assertEqual(gradients["WeightsGradient1"].shape,(1,2))
        self.assertEqual(gradients["BiasGradient1"].shape,(1,1))
        return parameters,gradients
    
    def test_updateWeights(self):
        parameters,gradients=self.test_backPropagate()
        oldParams=dict(parameters)
        updatedParams=self.network.UpdateWeights(parameters,gradients,10)
        self.assertTrue((updatedParams["bias1"]!=oldParams["bias1"]).all())
        self.assertTrue((updatedParams["Weights1"]!=oldParams["Weights1"]).all())
        
        
if __name__ == '__main__':
    unittest.main()