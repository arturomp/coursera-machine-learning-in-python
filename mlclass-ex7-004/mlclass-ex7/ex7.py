#!/usr/bin/env python

# python adaptation of solved ex7.m
# 

#  K-Means Clustering
#
#  depends on
#
#     findClosestCentroids.py
#     computeCentroids.py
#     runkMeans.py
#     kMeansInitCentroids.py
#

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import findClosestCentroids as fcc
import computeCentroids as cc
import runkMeans as rkm
import kMeansInitCentroids as kmic

## ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm 
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you shoudl complete the code in the findClosestCentroids function. 
#
print('Finding closest centroids.\n')

# Load an example dataset that we will be using
mat = scipy.io.loadmat('ex7data2.mat')
X = mat["X"]

# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = np.array( [[3, 3], [6, 2], [8, 5]] )

# Find the closest centroids for the examples using the
# initial_centroids
idx = fcc.findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: \n')
print(' {:s}'.format( idx[:3] ))
# adjusted next string for python's 0-indexing
print('\n(the closest centroids should be 0, 2, 1 respectively)\n')

raw_input('Program paused. Press enter to continue.')

## ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
print('\nComputing centroids means.\n')

#  Compute means based on the closest centroids found in the previous part.
centroids = cc.computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: \n')
print(' {:s} '.format(centroids))
print('\n(the centroids should be\n')
print('   [ 2.428301 3.157924 ]\n')
print('   [ 5.813503 2.633656 ]\n')
print('   [ 7.119387 3.616684 ]\n\n')

raw_input('Program paused. Press enter to continue.')


## =================== Part 3: K-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  kMeans algorithm. In this part, you will run the K-Means algorithm on
#  the example dataset we have provided. 
#
print('\nRunning K-Means clustering on example dataset.\n\n')

# Load an example dataset
mat = scipy.io.loadmat('ex7data2.mat')
X = mat["X"]

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = rkm.runkMeans(X, initial_centroids, max_iters, True)
print('\nK-Means Done.\n')

raw_input('Program paused. Press enter to continue.')

## ============= Part 4: K-Means Clustering on Pixels ===============
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel on to it's closest centroid.
#  
#  You should now complete the code in kMeansInitCentroids.m
#

print('\nRunning K-Means clustering on pixels from an image.\n\n')

#  Load an image of a bird
mat = scipy.io.loadmat('bird_small.mat')
A = mat["A"]

A = A / 255.0 # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(img_size[0] * img_size[1], 3, order='F').copy()

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16 
max_iters = 10

# When using K-Means, it is important the initialize the centroids
# randomly. 
# You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kmic.kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = rkm.runkMeans(X, initial_centroids, max_iters)

raw_input('Program paused. Press enter to continue.')


## ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we 

print('\nApplying K-Means to compress an image.\n')

# Find closest cluster members
idx = fcc.findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx. 

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by it's index in idx) to the centroid value
X_recovered = centroids[idx,:]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3, order='F')

# Display the original image
plt.close()
plt.subplot(1, 2, 1)
plt.imshow(A) 
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title( 'Compressed, with {:d} colors.'.format(K) )
plt.show(block=False)


raw_input('Program paused. Press enter to continue.')

