import numpy as np

def estimateGaussian(X):
    #ESTIMATEGAUSSIAN This function estimates the parameters of a 
    #Gaussian distribution using the data in X
    #   [mu sigma2] = estimateGaussian(X), 
    #   The input X is the dataset with each n-dimensional data point in one row
    #   The output is an n-dimensional vector mu, the mean of the data set
    #   and the variances sigma^2, an n x 1 vector
    # 

    # Useful variables
    m, n = X.shape

    # You should return these values correctly
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the mean of the data and the variances
    #               In particular, mu(i) should contain the mean of
    #               the data for the i-th feature and sigma2(i)
    #               should contain variance of the i-th feature.
    #

    # estimating mu - at this point it is an n-column vector
    # mu = (1/m)*sum(X,1);
    mu = np.mean(X, axis=0)
    # turn into n-rows vector
    mu = mu.T

    # estimating sigma^2 = std.dev.
    # normalizes with 1/N, instead of with 1/(N-1) in formula std (x) = 1/(N-1) SUM_i (x(i) - mean(x))^2 
    # i.e. degrees of freedom = 0 (by default)
    sigma2 = np.var(X, axis=0)

    # turn into n-rows vector
    sigma2 = sigma2.T

    # =============================================================

    return mu, sigma2
