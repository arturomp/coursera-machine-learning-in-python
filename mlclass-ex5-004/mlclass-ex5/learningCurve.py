import numpy as np
import trainLinearReg as tlr
import linearRegCostFunction as lrcf

def learningCurve(X, y, Xval, yval, lambda_val):
    #LEARNINGCURVE Generates the train and cross validation set errors needed 
    #to plot a learning curve
    #   [error_train, error_val] = ...
    #       LEARNINGCURVE(X, y, Xval, yval, lambda_val) returns the train and
    #       cross validation set errors for a learning curve. In particular, 
    #       it returns two vectors of the same length - error_train and 
    #       error_val. Then, error_train(i) contains the training error for
    #       i examples (and similarly for error_val(i)).
    #
    #   In this function, you will compute the train and test errors for
    #   dataset sizes from 1 up to m. In practice, when working with larger
    #   datasets, you might want to do this in larger intervals.
    #

    # Number of training examples
    m = len(X)

    # You need to return these values correctly
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))

    for i in xrange(1,m+1):

        # define training variables for this loop
        X_train = X[:i]
        y_train = y[:i]

        # learn theta parameters with current X_train and y_train
        theta = tlr.trainLinearReg(X_train, y_train, lambda_val)

        # fill in error_train(i) and error_val(i)
        #   note that for error computation, we set lambda_val = 0 in the last argument
        error_train[i-1] = lrcf.linearRegCostFunction(X_train, y_train, theta, 0)
        error_val[i-1]   = lrcf.linearRegCostFunction(Xval   , yval   , theta, 0)
                
    return error_train, error_val

