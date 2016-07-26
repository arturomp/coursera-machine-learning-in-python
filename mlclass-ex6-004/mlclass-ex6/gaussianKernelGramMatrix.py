import numpy as np
import gaussianKernel as gk

# from @lejlot http://stackoverflow.com/a/26962861/583834
def gaussianKernelGramMatrix(X1, X2, K_function=gk.gaussianKernel, sigma=0.1):
    """(Pre)calculates Gram Matrix K"""

    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = K_function(x1, x2, sigma)
    return gram_matrix
