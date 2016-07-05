import numpy as np
import matplotlib.pyplot as plt
import polyFeatures as pf

def plotFit(min_x, max_x, mu, sigma, theta, p):
    #PLOTFIT Plots a learned polynomial regression fit over an existing figure.
    #Also works with linear regression.
    #   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    #   fit with power p and feature normalization (mu, sigma).

    # Hold on to the current figure
    plt.hold(True)

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.array(np.arange(min_x - 15, max_x + 25, 0.05)) # 1D vector

    # Map the X values 
    X_poly = pf.polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly/sigma

    # Add ones
    X_poly = np.column_stack((np.ones((x.shape[0],1)), X_poly))

    # Plot
    plt.plot(x, np.dot(X_poly, theta), '--', linewidth=2)

    # Hold off to the current figure
    plt.hold(False)

