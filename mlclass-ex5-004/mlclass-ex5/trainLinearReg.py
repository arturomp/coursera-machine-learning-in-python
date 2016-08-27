import numpy as np
import linearRegCostFunction as lrcf
from scipy.optimize import minimize


def trainLinearReg(X, y, lambda_val):
    #TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    #regularization parameter lambda_val
    #   [theta] = TRAINLINEARREG (X, y, lambda_val) trains linear regression using
    #   the dataset (X, y) and regularization parameter lambda_val. Returns the
    #   trained parameters theta.
    #

    # Initialize Theta
    initial_theta = np.zeros((X.shape[1], 1))

    # Short hand for cost function to be minimized
    def costFunc(theta):
        return lrcf.linearRegCostFunction(X, y, theta, lambda_val, True)

    # Now, costFunction is a function that takes in only one argument
    maxiter = 200
    results = minimize(costFunc, x0=initial_theta, options={'disp': True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)

    theta = results["x"]

    return theta