#!/usr/bin/env python

# python adaptation of solved ex1_multi.m
# 
# Logistic Regression with multiple variables and regularization
# 
# depends on 
#     costFunctionReg.py
#     predict.py
#     plotData.py
#     mapFeature.py
#     plotDecisionBoundary.py

## Initialization

import costFunctionReg as cfr
import predict as pr
import plotData as pd
import mapFeature as mf
import plotDecisionBoundary as pdb
import numpy as np 
from scipy.optimize import fmin_bfgs

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data2.txt', delimiter=",")
X = data[:,:2]
y = data[:,2]

plt, p1, p2 = pd.plotData(X, y)

# # Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend((p1, p2), ('y = 1', 'y = 0'), numpoints=1, handlelength=0)

plt.show(block=False) # prevents having to close the graph to move forward with ex2_reg.py

raw_input('Program paused. Press enter to continue.\n')


## =========== Part 1: Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic 
#  regression to classify the data points. 
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mf.mapFeature(X[:,0], X[:,1])
m,n = X.shape

# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1
lambda_reg = 0.1

# Compute and display initial cost 
# gradient is too large to display in this exercise
cost = cfr.costFunctionReg(initial_theta, X, y, lambda_reg)

print('Cost at initial theta (zeros): {:f}'.format(cost))
# print('Gradient at initial theta (zeros):')
# print(grad)

raw_input('Program paused. Press enter to continue.\n')


## ============= Part 2: Regularization and Accuracies =============
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and 
#  see how regularization affects the decision coundart
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?

# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1 (you should vary this)
lambda_reg = 1

#  Run fmin_bfgs to obtain the optimal theta
#  This function returns theta and the cost 
myargs=(X, y, lambda_reg)
theta = fmin_bfgs(cfr.costFunctionReg, x0=initial_theta, args=myargs)

# Plot Boundary
pdb.plotDecisionBoundary(theta, X, y)

# # Labels, title and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.title('lambda = {:f}'.format(lambda_reg))

# % Compute accuracy on our training set
p = pr.predict(theta, X)

print('Train Accuracy: {:f}'.format(np.mean(p == y) * 100))

raw_input('Program paused. Press enter to continue.\n')

