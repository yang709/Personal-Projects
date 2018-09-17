# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 2018

@author: Xuan Yang

REF:
    https://www.nnwj.de/supervised-unsupervised.html
    https://www.youtube.com/watch?v=-7scQpJT7uo
    https://medium.com/@uniqtech/understand-the-softmax-function-in-minutes-f3a59641e86d
    https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
"""
import numpy as np
"""
    Neural Network:
        type:
            Supervised: if target output is provided, used for pattern association
            Unsupervised: no target output. used for classification
            
        Parameters:
            inLayer: number of nodes in the input layer
            outLayer: number of nodes in the out layer
            hiddenLayers(optional): an array represent the structure of the hiddenlayers
            actFunction(optional): choose the activation function to use.
            iniMethod(optional): initialization method for the weight matrix
            
        core idea:
            weights:
                Modify the value form previouse layer and pass it to the next layer.
                It direcly affect the output of the neural network.
                The training process is basiclly the process of modifying weights to make the output as accurate as possible.
                
            activation functions:
                The purpose of activation function is to make the input of each layer non-linear thus make multiple layer usable.
                
            forward propagation:
                The process of passing input through weights and hidden layers to the output.
                
                current_layer = activation_function(previouse_layer * weight)
                
            cost(error):
                the difference between the output of the neural network and the expected output.
                the smaller the cost is, the more accurate is the neural network.
                
            cost function: 
                The function used to calculate the cost(error) of the output of current training data(s).
                
                cost = expected_output - output
                
            gradient descent:
                Find the minimum of the cost function.
                
            backpropagation:
                back propagate the error to previous layer and modify connecting weights 
                beased on how much they contribute to the error .
        
                slope = derivative_of_ativation_function(current_layer)
                delta = cost * slope
                weight = weight + (previouse_layer.T * delta) * learning_rate
                bias = bias + delta * learning rate
        Algorithm:
            1. generate weight matrix based on layer (build the network)
            2. loop till end of trainig data:
                1. get current set of data
                2. loop for number of iteration:
                    1. forward propagation to get output
                    2. calculate the cost(error)
                    3. backpropagate the error to update weights and bias
                    
        side notes:
            weight initialization:
                
            activation functions:
               ReLU is currently most used and most accurate activation function 
               
"""

DEFAULT = object()

class NeuralNetwork:
    def __init__(self, inLayer, outLayer, hiddenLayers=[4,4], act_func = "ReLU", iniMethod = "he-et-al"):
        self.inLayer = inLayer
        self.outLayer = outLayer
        self.hiddenLayers = hiddenLayers
        self.act_func = act_func
        
        #create and initialize weight matrix
        self.weightMatrixs = []
        np.asarray(self.weightMatrixs)
        self.layer = [self.inLayer] + self.hiddenLayers + [self.outLayer]
        for i in range(len(self.layer)):
            if i == 0:
                continue
            if iniMethod == "he-et-al":
                weight = (np.random.randn(self.layer[i - 1],self.layer[i]) * np.sqrt(2/self.layer[i - 1]))
            elif iniMethod == "random":
                weight = (np.random.randn(self.layer[i - 1],self.layer[i]) * 0.01)
            else:
                #TODO： no such weight init method, return error
                return
            self.weightMatrixs.append(weight)
            
        #create and initialize bias matrix
        #all bias are initialized to zero
        self.bias = []
        for i in range(len(self.layer)):
            b = np.zeros(self.layer[i], np.int8).tolist()
            self.bias.append(b)
        
    """
    all activation functions
    
    """
    def activate(self, v, deri = False, last = False):
        if self.act_func == "Sig":
            if deri:
                return [value * (1 - value) for value in v]
            return [1 / (1 + np.exp(-value)) for value in v]
        elif self.act_func == "Tanh":
            if deri:
                return [1 - (value ** 2) for value in v]
            return [np.tanh(value) for value in v]
        elif self.act_func == "ReLU":
            return self.relu(v, deri, last)
        elif self.act_func == "softmax":
            return self.softmax(v, deri)
        else:
            print("no such activation function")
            return
           
    
    def relu(self, v, deri = False, last = False):
        if last:
            return self.softmax(v, deri)
        else:
            if np.any(np.isnan(v)):
                print("=====================")
                print(v)
            if deri:
                return [(value > 0) * 1 for value in v]
            return [np.maximum(0, value) for value in v]
        
    def softmax(self, v, deri = False):
        #if deri:
            
        exps = [np.exp(value - np.max(v)) for value in v]
        if np.any( np.isnan(exps)):
            print("   -----------------  ")
            print(v)
        denominator = sum(exps)
        return [exp / denominator for exp in exps]
    
    """
    forward propagation
    
    """
    def forward(self, input_values):
        self.layer_values = []                      #output value for each layer
        
        current_values = input_values               #value at current layer
        self.layer_values.append(current_values)    
        for i in range(len(self.layer)):            
            if i == 0:                              
                continue
            l = False
            if i == len(self.layer) - 1 and self.act_func == "ReLU":
                l = True
            current_values = self.activate(np.add(np.dot(current_values, self.weightMatrixs[i - 1]), self.bias[i]), last = l)
            #print(current_values)
            self.layer_values.append(current_values)
        
           
    """
    backpropagation
    
    """
    def backpropagation(self, exp_outputs, lr):
        #calculate the cost
        if len(exp_outputs[0]) != self.layer[-1]:
            print("expected output error")
        error = np.array(exp_outputs) - np.array(self.layer_values[-1]) 
        cost = error
        #totalCost = np.sum(cost, axis = 0)
        #print(totalCost)
        #backpropagate
        for i in reversed(range(len(self.layer))):
            if i == 0:
                break
            l = False
            if i == len(self.layer) - 1:
                l = True
            current_layer_slope = self.activate(self.layer_values[i], True, l)
            current_layer_delta = [cost[j] * current_layer_slope[j] for j in range(cost.shape[0])]
            cost = np.dot(current_layer_delta, self.weightMatrixs[i - 1].T)
            
            update = np.dot(np.array(self.layer_values[i - 1]).T, current_layer_delta) * lr
            self.weightMatrixs[i - 1] += update
            self.bias[i] += np.sum(current_layer_delta, axis = 0)*lr
            #print("-----------------------")

    """
    main training method:
    
    """
    def train(self, input_values, exp_outputs = DEFAULT, iterations = 1000, learning_rate = 0.02):
        #self.forward(input_values)
        
        for ite in range(iterations):
            if exp_outputs is DEFAULT:
                print("no expected output provided")
                return
            
            #forward propagation
            self.forward(input_values)
                
            #backpropagation
            self.backpropagation(exp_outputs, learning_rate)
        
        print(self.layer_values[-1])

            
            
            
            