#!/usr/bin/env python

# python adaptation of solved ex7_pca.m
# 

#  Principle Component Analysis and K-Means Clustering
#
#  depends on
#
#     featureNormalize.py
#     pca.py
#     drawLine
#     projectData.py
#     recoverData.py
#     displayData.py
#     kMeansInitCentroids.py
#     runkMeans.py
#     hsv.py
#     plotDataPoints.py
#

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import featureNormalize as fn
import pca
import drawLine as dl
import projectData as pd
import recoverData as rd
import displayData as dd
import kMeansInitCentroids as kmic
import runkMeans as rkm
import hsv 
from mpl_toolkits.mplot3d import Axes3D
import plotDataPoints as pdp

## ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize
#
print('Visualizing example dataset for PCA.\n');

#  The following command loads the dataset. You should now have the 
#  variable X in your environment
mat = scipy.io.loadmat('ex7data1.mat')
X = np.array(mat["X"])

# interactive graphs
plt.ion()

#  Visualize the example dataset
plt.close()

# kept the scatter() (vs. the plot()) version 
#  because scatter() makes properly circular markers
# plt.plot(X[:, 0], X[:, 1], 'o', markersize=9, markeredgewidth=1, markeredgecolor='b', markerfacecolor='None')
plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors='b')
plt.axis([0.5, 6.5, 2, 8])
plt.gca().set_aspect('equal', adjustable='box')
plt.show(block=False)


raw_input('Program paused. Press enter to continue.')


## =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('Running PCA on example dataset.\n');

#  Before running PCA, it is important to first normalize X
X_norm, mu, _ = fn.featureNormalize(X)

#  Run PCA
U, S = pca.pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.
plt.hold(True)
dl.drawLine(mu, mu + 1.5 * S[0,0] * U[:,0].T, c='k', linewidth=2)
dl.drawLine(mu, mu + 1.5 * S[1,1] * U[:,1].T, c='k', linewidth=2)
plt.hold(False)

print('Top eigenvector: \n')
print(' U(:,1) = {:f} {:f} \n'.format(U[0,0], U[1,0]))
print('(you should expect to see -0.707107 -0.707107)')


raw_input('Program paused. Press enter to continue.')


## =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the 
#  first k eigenvectors. The code will then plot the data in this reduced 
#  dimensional space.  This will show you what the data looks like when 
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m
#
print('Dimension reduction on example dataset.\n');

#  Plot the normalized dataset (returned from pca)
plt.close()
plt.scatter(X_norm[:,0], X_norm[:,1], s=75, facecolors='none', edgecolors='b')
plt.axis([-4, 3, -4, 3])
plt.gca().set_aspect('equal', adjustable='box')
plt.show(block=False)

#  Project the data onto K = 1 dimension
K = 1
Z = pd.projectData(X_norm, U, K)
print('Projection of the first example: {:s}\n'.format(Z[0]))
print('(this value should be about 1.481274)\n')

X_rec  = rd.recoverData(Z, U, K)
print('Approximation of the first example: {:f} {:f}\n'.format(X_rec[0, 0], X_rec[0, 1]))
print('(this value should be about  -1.047419 -1.047419)\n')

#  Draw lines connecting the projected points to the original points
plt.hold(True)
plt.scatter(X_rec[:, 0], X_rec[:, 1], s=75, facecolors='none', edgecolors='r')
for i in xrange(X_norm.shape[0]):
    dl.drawLine(X_norm[i,:], X_rec[i,:], linestyle='--', color='k', linewidth=1)
plt.hold(False)


raw_input('Program paused. Press enter to continue.')


## =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('Loading face dataset.\n');

#  Load Face dataset
mat = scipy.io.loadmat('ex7faces.mat')
X = np.array(mat["X"])

#  Display the first 100 faces in the dataset
dd.displayData(X[:100, :])

raw_input('Program paused. Press enter to continue.')

## =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#
print('Running PCA on face dataset.\n(this mght take a minute or two ...)\n')

#  Before running PCA, it is important to first normalize X by subtracting 
#  the mean value from each feature
X_norm, _, _ = fn.featureNormalize(X)

#  Run PCA
U, S = pca.pca(X_norm)

#  Visualize the top 36 eigenvectors found
dd.displayData(U[:, :36].T)


raw_input('Program paused. Press enter to continue.')


## ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors 
#  If you are applying a machine learning algorithm 
print('Dimension reduction for face dataset.\n');

K = 100
Z = pd.projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print('{:d} {:d}'.format(Z.shape[0], Z.shape[1]))


raw_input('Program paused. Press enter to continue.')


## ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and 
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('Visualizing the projected (reduced dimension) faces.\n');

K = 100;
X_rec  = rd.recoverData(Z, U, K)

# Display normalized data
plt.close()
plt.subplot(1, 2, 1)
dd.displayData(X_norm[:100,:])
plt.title('Original faces')
plt.gca().set_aspect('equal', adjustable='box')

# Display reconstructed data from only k eigenfaces
plt.subplot(1, 2, 2)
dd.displayData(X_rec[:100,:])
plt.title('Recovered faces')
plt.gca().set_aspect('equal', adjustable='box')


raw_input('Program paused. Press enter to continue.')


## === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

plt.close()

# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first

# A = double(imread('bird_small.png'));
mat = scipy.io.loadmat('bird_small.mat')
A = mat["A"]

# from ex7.py, part 4
A = A / 255.0
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], 3, order='F').copy()
K = 16 
max_iters = 10
initial_centroids = kmic.kMeansInitCentroids(X, K)
centroids, idx = rkm.runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
#  use flatten(). otherwise, Z[sel, :] yields array w shape [1000,1,2]
sel = np.floor(np.random.rand(1000, 1) * X.shape[0]).astype(int).flatten()

#  Setup Color Palette
palette = hsv.hsv(K)
colors = np.array([palette[int(i)] for i in idx[sel]])

#  Visualize the data and centroid memberships in 3D
fig1 = plt.figure(1)
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=100, c=colors)
plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show(block=False)

raw_input('Program paused. Press enter to continue.\n\n')

## === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
X_norm, _, _ = fn.featureNormalize(X)

# PCA and project the data to 2D
U, S = pca.pca(X_norm)
Z = pd.projectData(X_norm, U, 2)

# Plot in 2D
fig2 = plt.figure(2)
pdp.plotDataPoints(Z[sel, :], idx[sel], K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
plt.show(block=False)

raw_input('Program paused. Press enter to continue.')
