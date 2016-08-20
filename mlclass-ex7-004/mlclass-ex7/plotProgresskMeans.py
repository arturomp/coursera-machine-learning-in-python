import plotDataPoints as pdp
import drawLine as dl
import matplotlib.pyplot as plt

def plotProgresskMeans(X, centroids, previous, idx, K, i):
    #PLOTPROGRESSKMEANS is a helper function that displays the progress of 
    #k-Means as it is running. It is intended for use only with 2D data.
    #   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
    #   points with colors assigned to each centroid. With the previous
    #   centroids, it also plots a line between the previous locations and
    #   current locations of the centroids.
    #

    # plt.hold(True)

    # Plot the examples
    pdp.plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=400, c='k', linewidth=1)

    # Plot the history of the centroids with lines
    for j in xrange(centroids.shape[0]):
        dl.drawLine(centroids[j, :], previous[j, :], c='b')

    # Title
    plt.title('Iteration number {:d}'.format(i+1))

    return

