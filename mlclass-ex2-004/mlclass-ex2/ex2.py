#!/usr/bin/env python

# python adaptation of solved ex1_multi.m
# 
# Linear regression with multiple variables
# 
# depends on 
#   - warmUpExercise.py
#   - ex1data1.txt
#   - plotData.py
#   - computeCost.py
#   - gradientDescent.py

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

plt.hold(False) # prevents further drawing on plot
plt.show(block=False) # prevents having to close the graph to move forward with ex2.py

raw_input('Program paused. Press enter to continue.\n')


