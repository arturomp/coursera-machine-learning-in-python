import numpy as np
import scipy.linalg as linalg

def pca(X):
    #PCA Run principal component analysis on the dataset X
    #   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
    #   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
    #

    # Useful values
    m, n = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the "svd" function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix. 
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    #

    # compute the covariance matrix
    sigma = (1.0/m) * (X.T).dot(X)

    # compute the eigenvectors (U) and S
    # from: 
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html#scipy.linalg.svd
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.diagsvd.html#scipy.linalg.diagsvd
    U, S, Vh = linalg.svd(sigma)
    S = linalg.diagsvd(S, len(S), len(S))

    # =========================================================================

    return U, S
