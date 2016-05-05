#!/usr/bin/env python

# python adaptation of solved ex3.m
# 
# Multiple class classification
# 
# depends on 
#
#     displayData.py
#     lrCostFunction.py (logistic regression cost function)
#     oneVsAll.py
#     predictOneVsAll.py
#     predict.py
#

import scipy.io
import numpy as np
import displayData as dd
import oneVsAll as ova
import predictOneVsAll as pova

## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('ex3data1.mat')

X = mat["X"]
y = mat["y"]

m = X.shape[0]

# crucial step in getting good performance!
# changes the dimension from (m,1) to (m,)
# otherwise the minimization isn't very effective...
y=y.flatten() 

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100],:]

dd.displayData(sel)

raw_input('Program paused. Press enter to continue.\n')

## ============ Part 2: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

print('Training One-vs-All Logistic Regression...')

lambda_reg = 0.1
all_theta = ova.oneVsAll(X, y, num_labels, lambda_reg)

raw_input('Program paused. Press enter to continue.\n')

## ================ Part 3: Predict for One-Vs-All ================
#  After ...
pred = pova.predictOneVsAll(all_theta, X)

print('Training Set Accuracy: {:f}'.format((np.mean(pred == y%10)*100)))
print('Training Set Accuracy for 1:  {:f}'.format(np.mean(pred[500:1000]  == y.flatten()[500:1000]%10)  * 100))
print('Training Set Accuracy for 2:  {:f}'.format(np.mean(pred[1000:1500] == y.flatten()[1000:1500]%10) * 100))
print('Training Set Accuracy for 3:  {:f}'.format(np.mean(pred[1500:2000] == y.flatten()[1500:2000]%10) * 100))
print('Training Set Accuracy for 4:  {:f}'.format(np.mean(pred[2000:2500] == y.flatten()[2000:2500]%10) * 100))
print('Training Set Accuracy for 5:  {:f}'.format(np.mean(pred[2500:3000] == y.flatten()[2500:3000]%10) * 100))
print('Training Set Accuracy for 6:  {:f}'.format(np.mean(pred[3000:3500] == y.flatten()[3000:3500]%10) * 100))
print('Training Set Accuracy for 7:  {:f}'.format(np.mean(pred[3500:4000] == y.flatten()[3500:4000]%10) * 100))
print('Training Set Accuracy for 8:  {:f}'.format(np.mean(pred[4000:4500] == y.flatten()[4000:4500]%10) * 100))
print('Training Set Accuracy for 9:  {:f}'.format(np.mean(pred[4500:5000] == y.flatten()[4500:5000]%10) * 100))
print('Training Set Accuracy for 10: {:f}'.format(np.mean(pred[0:500]     == y.flatten()[0:500]%10)     * 100))
