# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#input data
X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
#output data
y = np.array([[1],[1],[0]])
9715
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def der_sig(x):
    return x*(1-x)
#iteration number
ite_count = 5000
#learning rate
lr = 0.1
#initialize layers size
inputlayer_size = X.shape[1]
hiddenlayer_size = 3
outputlayer_size = y.shape[1]

#initialize weight and bias of links
weight_in=np.random.uniform(size=(inputlayer_size, hiddenlayer_size))
weight_out=np.random.uniform(size=(hiddenlayer_size, outputlayer_size))
bias_in = np.random.uniform(size=(1,hiddenlayer_size))
bias_out = np.random.uniform(size=(1,outputlayer_size))

for i in range(ite_count):
    
    #Forward Propogation
    #1. Get input of hidden layer by dot production of data and input weight. 
    hiddenlayer_input = np.dot(X,weight_in) + bias_in
    #2. Process in hidden layer use sigmoid function.
    hiddenlayer_result = sigmoid(hiddenlayer_input)
    #3. Forward to output layer
    outputlayer_input = np.dot(hiddenlayer_result, weight_out) + bias_out
    #4. Process the output use sigmoid function.
    output = sigmoid(outputlayer_input)
    
    #Back Propagation
    #1. Calc error.
    error = y - output
    #2. Calc the slope(derivative) of each layer.
    #(how much are each value contribute to the error)
    outputlayer_slope = der_sig(output)
    hiddenlayer_slope = der_sig(hiddenlayer_result)
    #3 Backpropagate the error to previous layer.
    #3.1 Devide error to output layer, get the change factor.
    outputlayer_change = error * outputlayer_slope
    #3.2 get the error on hidden layer.
    hiddenlayer_error = outputlayer_change.dot(weight_out.T)
    #3.3 Hidden layer change factor
    hiddenlayer_change = hiddenlayer_error * hiddenlayer_slope
    #4 update weights and bias
    weight_out = weight_out + hiddenlayer_result.T.dot(outputlayer_change) * lr
    bias_out = bias_out + np.sum(outputlayer_change, axis = 0, keepdims = True)
    weight_in = weight_in + X.T.dot(hiddenlayer_change) * lr
    bias_in = bias_in + np.sum(hiddenlayer_change, axis = 0, keepdims = True)

print(output)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    