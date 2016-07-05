import numpy as np

def polyFeatures(X, p):
    #POLYFEATURES Maps X (1D vector) into the p-th power
    #   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
    #   maps each example into its polynomial features where
    #   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    #


    # You need to return the following variables correctly.
    # X_poly = np.zeros((X.size, p))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Given a vector X, return a matrix X_poly where the p-th 
    #               column of X contains the values of X to the p-th power.
    #
    # 

    # initialize X_poly to be equal to the single-column X
    X_poly = X


    # if p is equal or greater than 2
    if p >= 2:

        # for each number between column 2 (index 1) and last column
        for k in xrange(1,p):

            # add k-th column of polynomial features where k-th column is X.^k
            X_poly = np.column_stack((X_poly, np.power(X,k+1)))
            

    return X_poly

    # =========================================================================
