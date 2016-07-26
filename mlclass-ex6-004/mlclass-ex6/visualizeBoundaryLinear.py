import numpy as np
import matplotlib.pyplot as plt
import plotData as pd

def visualizeBoundaryLinear(X, y, model):
    #VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
    #SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
    #   learned by the SVM and overlays the data on it

    # plot decision boundary
    # right assignments from http://stackoverflow.com/a/22356267/583834
    w = model.coef_[0]
    b = model.intercept_[0]
    xp = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    yp = - (w[0] * xp + b) / w[1]

    plt.plot(xp, yp, 'b-')

    # plot training data
    pd.plotData(X, y)
