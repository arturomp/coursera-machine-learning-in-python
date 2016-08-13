import sys
import numpy as np
import matplotlib.pyplot as plt
import computeCentroids as cc
import findClosestCentroids as fcc
import plotProgresskMeans as ppkm

def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    #RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    #is a single example
    #   [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
    #   plot_progress) runs the K-Means algorithm on data matrix X, where each 
    #   row of X is a single example. It uses initial_centroids used as the
    #   initial centroids. max_iters specifies the total number of interactions 
    #   of K-Means to execute. plot_progress is a true/false flag that 
    #   indicates if the function should also plot its progress as the 
    #   learning happens. This is set to false by default. runkMeans returns 
    #   centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
    #   vector of centroid assignments (i.e. each entry in range [1..K])
    #

    # Set default value for plot progress
    # (commented out due to pythonic default parameter assignment above)
    # if not plot_progress:
    #     plot_progress = False

    # Plot the data if we are plotting progress
    # if plot_progress:
    #     plt.hold(True)

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))

    # if plotting, set up the space for interactive graphs
    # http://stackoverflow.com/a/4098938/583834
    # http://matplotlib.org/faq/usage_faq.html#what-is-interactive-mode
    if plot_progress:
        plt.close()
        plt.ion()

    # Run K-Means
    for i in xrange(max_iters):
        
        # Output progress
        sys.stdout.write('\rK-Means iteration {:d}/{:d}...'.format(i+1, max_iters))
        sys.stdout.flush()
        
        # For each example in X, assign it to the closest centroid
        idx = fcc.findClosestCentroids(X, centroids)
        
        # Optionally, plot progress here
        if plot_progress:
            ppkm.plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            raw_input('Press enter to continue.')
        
        # Given the memberships, compute new centroids
        centroids = cc.computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    print('\n')

    # if plot_progress:
    #     plt.hold(False)

    return centroids, idx

