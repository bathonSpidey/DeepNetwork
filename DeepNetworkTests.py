# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:21:42 2021

@author: batho
"""
import DeepNetwork
import unittest

class TestNetwork(unittest.TestCase):
    
    def test_InitilaizeNetworkWithInvalidInput(self):
        network=DeepNetwork.Network()
        with self.assertRaises(Exception):network.InitializeNetwork("1,2,4,3")
        with self.assertRaises(Exception):network.InitializeNetwork([])
        with self.assertRaises(Exception):network.InitializeNetwork([0,2,1])
        
    def test_GetParameters(self):
        network=DeepNetwork.Network()
        parameters=network.InitializeNetwork([2,2,1])
        self.assertEqual(len(parameters.keys()),4)
        self.assertEqual(parameters["Weights1"].shape,(2,2))
        self.assertEqual(parameters["Weights2"].shape,(1,2))
        self.assertEqual(parameters["bias1"].shape,(2,1))
        self.assertEqual(parameters["bias2"].shape,(1,1))
        
    def test_GetParamtersOfDeeperNetwork(self):
        network=DeepNetwork.Network()
        parameters=network.InitializeNetwork([2,8,2,6,1])
        self.assertEqual(len(parameters.keys()),8)
        self.assertEqual(parameters["Weights1"].shape,(8,2))
        self.assertEqual(parameters["Weights2"].shape,(2,8))
        self.assertEqual(parameters["Weights3"].shape,(6,2))
        self.assertEqual(parameters["Weights4"].shape,(1,6))
        self.assertEqual(parameters["bias1"].shape,(8,1))
        self.assertEqual(parameters["bias2"].shape,(2,1))
    
    
    
        


if __name__ == '__main__':
    unittest.main()