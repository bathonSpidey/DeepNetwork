# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 23:01:33 2021

@author: batho
"""

import unittest

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
        
    
    

class TestNetwork(unittest.TestCase):
    
    def test_InitilaizeNetworkWithInvalidInput(self):
        network=Network()
        with self.assertRaises(Exception):network.InitializeNetwork("1,2,4,3")
        with self.assertRaises(Exception):network.InitializeNetwork([])
        with self.assertRaises(Exception):network.InitializeNetwork([0,2,1])
        
    def test_GetParameters(self):
        network=Network()
        parameters=network.InitializeNetwork([2,2,1])
        self.assertEqual(len(parameters.keys()),4)
        self.assertEqual(parameters["Weights1"].shape,(2,2))
        self.assertEqual(parameters["Weights2"].shape,(1,2))
        self.assertEqual(parameters["bias1"].shape,(2,1))
        self.assertEqual(parameters["bias2"].shape,(1,1))
        
        
    
    
    
        


if __name__ == '__main__':
    unittest.main()