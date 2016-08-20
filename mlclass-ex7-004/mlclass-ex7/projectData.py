import numpy as np

def projectData(X, U, K):
    #PROJECTDATA Computes the reduced data representation when projecting only 
    #on to the top k eigenvectors
    #   Z = projectData(X, U, K) computes the projection of 
    #   the normalized inputs X into the reduced dimensional space spanned by
    #   the first K columns of U. It returns the projected examples in Z.
    #

    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the projection of the data using only the top K 
    #               eigenvectors in U (first K columns). 
    #               For the i-th example X(i,:), the projection on to the k-th 
    #               eigenvector is given as follows:
    #                    x = X(i, :)';
    #                    projection_k = x' * U(:, k);
    #

    # get U_reduce for only the desired K
    U_reduce = U[:,:K]

    # get Z - the projections from X onto the space defined by U_reduce
    #	note that this vectorized version performs the projection the instructions
    # 	above but in one operation
    Z = X.dot(U_reduce)

    # =============================================================

    return Z