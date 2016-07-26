import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    #PLOTDATA Plots the data points X and y into a new figure 
    #   PLOTDATA(x,y) plots the data points with + for the positive examples
    #   and o for the negative examples. X is assumed to be a Mx2 matrix.
    #
    # Note: This was slightly modified such that it expects y = 1 or y = 0

    # Find Indices of Positive and Negative Examples
    y = y.flatten()
    pos = y==1
    neg = y==0

    # Plot Examples
    plt.plot(X[:,0][pos], X[:,1][pos], "k+", markersize=10)
    plt.plot(X[:,0][neg], X[:,1][neg], "yo", markersize=10)

    # alternatives
    #
    # plt.scatter(*zip(*X[pos]), c="k", marker='+', s=100)
    # plt.scatter(*zip(*X[neg]), c="y", marker='o', s=100)
    #
    # plt.plot(X[:,0][pos], X[:,1][pos], 'k+', X[:,0][neg], X[:,1][neg], 'yo', markersize=10)

    plt.show(block=False) 
