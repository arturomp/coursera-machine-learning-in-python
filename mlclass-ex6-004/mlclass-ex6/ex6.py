#!/usr/bin/env python

# python adaptation of solved ex6.m
# 

#  Support Vector Machines (SVMs)
#
#  depends on
#
#     plotData.py
#     svmTrain.py
#     visualizeBoundaryLinear.py
#     gaussianKernel.py
#     gaussianKernelGramMatrix.py
#     visualizeBoundary.py
#     dataset3Params.py
#

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import plotData as pd
import svmTrain as svmt
import visualizeBoundaryLinear as vbl
import gaussianKernel as gk
import visualizeBoundary as vb
import dataset3Params as dp


## =============== Part 1: Loading and Visualizing Data ================
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

print('Loading and Visualizing Data ...')

# Load from ex6data1: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data1.mat')
X = mat["X"]
y = mat["y"]

pd.plotData(X, y)

raw_input('Program paused. Press enter to continue.')

## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.
#

# Load from ex6data1: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data1.mat')
X = mat["X"]
y = mat["y"]

print('Training Linear SVM ...')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = svmt.svmTrain(X, y, C, "linear", 1e-3, 20)
plt.close()
vbl.visualizeBoundaryLinear(X, y, model)

raw_input('Program paused. Press enter to continue.')

## =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('Evaluating the Gaussian Kernel ...')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gk.gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 2 :' \
         '\n\t{:f}\n(this value should be about 0.324652)\n'.format(sim))

raw_input('Program paused. Press enter to continue.')

## =============== Part 4: Visualizing Dataset 2 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...')

# Load from ex6data2: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data2.mat')
X = mat["X"]
y = mat["y"]

# Plot training data
plt.close()
pd.plotData(X, y)

raw_input('Program paused. Press enter to continue.')

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the 
#  SVM classifier.
# 
print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...');

# Load from ex6data2: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data2.mat')
X = mat["X"]
y = mat["y"]

# SVM Parameters
C = 1
sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.
model = svmt.svmTrain(X, y, C, "gaussian")

# alternative if don't want to implement Gaussian kernel
# model = svmt.svmTrain(X, y.flatten(), C, "rbf", gamma=6) 

plt.close()
vb.visualizeBoundary(X, y, model)

raw_input('Program paused. Press enter to continue.')

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and 
#  plot the data. 
#

print('Loading and Visualizing Data ...')

# Load from ex6data3: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data3.mat')
X = mat["X"]
y = mat["y"]

# Plot training data
plt.close()
pd.plotData(X, y)

raw_input('Program paused. Press enter to continue.')

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
# 

# Load from ex6data3: 
# You will have X, y in your environment
mat = scipy.io.loadmat('ex6data3.mat')
X = mat["X"]
y = mat["y"]
Xval = mat["Xval"]
yval = mat["yval"]

# Try different SVM Parameters here
C, sigma = dp.dataset3Params(X, y, Xval, yval)

# train model on training corpus with current sigma and C
model = svmt.svmTrain(X, y, C, "gaussian", sigma=sigma)
plt.close()
vb.visualizeBoundary(X, y, model)

raw_input('Program paused. Press enter to continue.')

