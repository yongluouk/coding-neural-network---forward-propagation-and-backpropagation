# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:30:27 2019

@author: China Cache
"""
import numpy as np
np.random.seed(100)

def initialize(num_f, num_h1, num_h2, num_out):
    W_B = {
        'W1': np.random.randn(num_h1, num_f),
        'b1': np.random.randn(num_h1, 1),
        'W2': np.random.randn(num_h2, num_h1),
        'b2': np.random.randn(num_h2, 1),
        'W3': np.random.randn(num_out, num_h2),
        'b3': np.random.randn(num_out, 1)
    }
    return W_B

def relu(Z):
    return np.maximum(Z, 0)
def dRelu(Z):
    return 1 * (Z > 0) 

def sigmoid(Z):
    sigmoid_Z = 1 / (1 + (np.exp(-Z)))
    return sigmoid_Z
    
def softmax(scores):
  return np.exp(scores)/sum(np.exp(scores))

def forward_propagation(X, W_B):
    Z1 = np.matmul(W_B['W1'], X.T)  + W_B['b1']
    print(Z1)
    A1 = relu(Z1)
    print(sigmoid(Z1))
    Z2 = np.matmul(W_B['W2'], A1) + W_B['b2']
    A2 = relu(Z2)
    Z3 = np.matmul(W_B['W3'], A2) + W_B['b3']
    Y_pred = sigmoid(Z3.T)
    
    forward_results = {"Z1": Z1,
                      "A1": A1,
                      "Z2": Z2,
                      "A2": A2,
                      "Z3": Z3,
                      "Y_pred": Y_pred}
    
    return forward_results

def backward_propagation(X, W_B, Y_true):
    Z1 = np.matmul(W_B['W1'], X.T)  + W_B['b1']
    A1 = relu(Z1)
    Z2 = np.matmul(W_B['W2'], A1) + W_B['b2']
    #print(Z2)
    #print(Z2.T>0)
    A2 = relu(Z2)
    Z3 = np.matmul(W_B['W3'], A2) + W_B['b3']
    Y_pred = sigmoid(Z3.T)
    
    no_examples = X.shape[0]
    
    L = (1/no_examples) * \
        np.sum(-Y_true * np.log(Y_pred) - (1 - Y_true) * np.log(1 - Y_pred))
    
    dLdZ3 = (1/no_examples) * (Y_pred - Y_true)
    dLdW3 = np.matmul(A2, dLdZ3)
    dLdb3 = np.sum(dLdZ3, axis=0)
    dLdZ2 = np.matmul(dLdZ3, W_B['W3']) * (Z2.T > 0)
    #print(dLdZ2)
    #print(np.matmul(dLdZ3, W_B['W3']))
    dLdW2 = np.matmul(A1, dLdZ2)
    dLdb2 = np.sum(dLdZ2, axis=0)
    dLdZ1 = np.matmul(dLdZ2, W_B['W2']) * (Z1.T > 0)
    dLdW1 = np.matmul(X.T, dLdZ1)
    dLdb1 = np.sum(dLdZ1, axis=0)
    gradients = {"dLdW1": dLdW1,
                 "dLdb1": dLdb1,
                 "dLdW2": dLdW2,
                 "dLdb2": dLdb2,
                 "dLdW3": dLdW3,
                 "dLdb3": dLdb3}
    
    return gradients, L

X = np.array([[0,0],[0,1],[1,0],[1,1],[1,1]])
Y_true = np.array([[0],[0],[0],[1],[1]])
W_B = initialize(2,4,3,1)
#backward_results = backward_propagation(X, W_B, Y_true)
forward_results = forward_propagation(X, W_B)
    
#print(backward_results)
    
