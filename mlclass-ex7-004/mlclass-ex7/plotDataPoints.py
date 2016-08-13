import hsv
import matplotlib.pyplot as plt
import numpy as np

def plotDataPoints(X, idx, K):
    #PLOTDATAPOINTS plots data points in X, coloring them so that those with the same
    #index assignments in idx have the same color
    #   PLOTDATAPOINTS(X, idx, K) plots data points in X, coloring them so that those 
    #   with the same index assignments in idx have the same color

    # Create palette (see hsv.py)
    p = hsv.hsv( K )
    colors = np.array([p[int(i)] for i in idx])

    # Plot the data
    plt.scatter(X[:,0], X[:,1], s=75, facecolors='none', edgecolors=colors)

    return
