#!/usr/bin/env python

# python adaptation of solved ex1_multi.m
# 
# Linear regression with multiple variables
# 
# depends on 
#     sigmoid.py
#     costFunction.py
#     predict.py
#     costFunctionReg.py

## Initialization

import numpy as np 

## Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data1.txt', delimiter=",")
X = data[:,:2]
y = data[:,2]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

import plotData as pd

plt, p1, p2 = pd.plotData(X, y)

# # Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((p1, p2), ('Admitted', 'Not Admitted'), numpoints=1, handlelength=0)

plt.show(block=False) # prevents having to close the graph to move forward with ex2.py

raw_input('Program paused. Press enter to continue.\n')
# plt.close()

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m

#  Setup the data matrix appropriately, and add ones for the intercept term
m,n = X.shape
X_padded = np.column_stack((np.ones((m,1)), X)) 

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1));

# Compute and display initial cost and gradient
import costFunction as cf
# cost, grad = cf.costFunction(initial_theta, X_padded, y);
cost, grad = cf.costFunction(initial_theta, X_padded, y, return_grad=True);

print('Cost at initial theta (zeros): {:f}'.format(cost));
print('Gradient at initial theta (zeros):');
print(grad);

raw_input('Program paused. Press enter to continue.\n')


## ============= Part 3: Optimizing using fmin  =============
#  In this exercise, you will use a built-in function (fmin) to find the
#  optimal parameters theta.

from scipy.optimize import fmin
# from scipy.optimize import fmin_bfgs

#  Run fmin to obtain the optimal theta
#  This function will return theta and the cost 

myargs=(X_padded, y, False)
theta = fmin(cf.costFunction, x0=initial_theta, args=myargs)
# theta = fmin_bfgs(cf.costFunction, x0=initial_theta, args=(X_padded, y, return_grad=False), maxiter=400)

# Print theta to screen
cost_at_theta = cf.costFunction(theta, X_padded, y, False)
print('Cost at theta found by fmin: {:f}'.format(cost_at_theta))
print('theta:'),
print(theta)

# Plot Boundary
import plotDecisionBoundary as pdb
pdb.plotDecisionBoundary(theta, X, y);

plt.hold(False) # prevents further drawing on plot
plt.show(block=False) 

raw_input('Program paused. Press enter to continue.\n')

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of 
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

from sigmoid import sigmoid
from predict import predict
prob = sigmoid(np.dot(np.array([1,45,85]),theta));
print('For a student with scores 45 and 85, we predict an admission probability of {:f}'.format(prob));

# Compute accuracy on our training set
p = predict(theta, X_padded)

print('Train Accuracy: {:f}'.format(np.mean(p == y) * 100))

raw_input('Program paused. Press enter to continue.\n')
