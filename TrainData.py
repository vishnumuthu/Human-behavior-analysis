# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:54:46 2018

@author: vishnu
"""
#%%
import numpy as np
import math
# Setup the parameters you will use for this part of the exercise
input_layer_size  = 400   # 20x20 Input Images of Digits
num_labels = 10;          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

# Load Training Data
print("Loading and Visualizing Data ...\n")

#f = open(filename) # training data stored in arrays X, y

# Test case for lrCostFunction
print("\nTesting lrCostFunction() with regularization")

theta_t = np.array([[2.4014, .000000066011, -1.1799, -1.1799,-1.1799],[-1.8121, -.0000000356,0.58354,0.58354,0.58354],[-3.0399,0.00000030905,0.67869,0.67869,0.67870]])
X_t = np.array([[1 , .1, .6, 1.1],[1 , .2, .7, 1.2],[ 1, .3, .8, 1.3],[ 1, .4, .9, 1.4],[ 1, .5, 1, 1.5]])
y_t = np.array([[1],[0],[1],[0],[1]])
lambda_t = 3

#%%
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def lrCostFunction(theta, X, y, lambda_val):
    m = y.shape[0]  # number of training examples

    Cval = 0
    hval = np.zeros(m)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            hval[i]= hval[i] + X[i][j]*theta[j]
    
    for i in range(m):
        Cval = Cval + (-1* y[i] * math.log(sigmoid(hval[i])) \
               -1*(1-y[i])*(math.log(1-sigmoid(hval[i]))))
        
    Sval = 0
    [n,s] = theta.shape
    for i in range(1,n,1):
        Sval = Sval + theta[i]**2
    
    J = (1/m) * Cval + (lambda_val/(2*m))*Sval

    grad = np.zeros(theta.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
                grad[j] = grad[j] + (1/m) * ((sigmoid(hval[i]) - y[i]) * X[i][j])

    for i in range(1,X.shape[1]):
        grad[i] = grad[i] + (lambda_val/m) * (theta[i])
       
    return J,grad
#%%
def predictOneVsAll(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    # return variables 
    p = np.zeros(m)
    # Add ones to the X data matrix
    X = np.hstack([np.ones((m,1)), X])

    for i in range(m):
        Sval = np.zeros(num_labels)
        for j in range(num_labels):
            nval = 0
            for k in range(X.shape[1]):
                nval = nval + all_theta[j][k] * X[i][k]
            Sval[j] = sigmoid(nval)
        p[i] = np.argmax(Sval)
    print (p)
#%%


