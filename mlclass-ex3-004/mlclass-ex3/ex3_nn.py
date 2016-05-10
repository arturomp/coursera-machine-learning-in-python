#!/usr/bin/env python

# python adaptation of solved ex3_nn.m
# 
# Exercise 3 | Part 2: Neural Networks
# 
# depends on 
#
#     displayData.py
#     predict.py
#

import numpy as np
import displayData as dd
import predict as pr

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

import scipy.io
mat = scipy.io.loadmat('ex3data1.mat')

X = mat["X"]
y = mat["y"]

y = y.flatten()

m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100],:]

dd.displayData(sel)

raw_input('Program paused. Press enter to continue.\n')

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
mat = scipy.io.loadmat('ex3weights.mat')
Theta1 = mat["Theta1"]
Theta2 = mat["Theta2"]

## ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = pr.predict(Theta1, Theta2, X)

print('Training Set Accuracy: {:f}'.format((np.mean(pred == y)*100)))

raw_input('Program paused. Press enter to continue.\n')

#  To give you an idea of the network's output, you can also run
#  through the examples one at a time to see what it is predicting.

#  Randomly permute examples
rp = np.random.permutation(m)

for i in xrange(m):

    # Display 
    print('Displaying Example Image')
    dd.displayData(X[rp[i], :])

    pred = pr.predict(Theta1, Theta2, X[rp[i], :])
    print('Neural Network Prediction: {:d} (digit {:d})'.format(pred[0], (pred%10)[0]))
    
    raw_input('Program paused. Press enter to continue.\n')
