def sigmoid(z):
#SIGMOID Compute sigmoid functoon
#   J = SIGMOID(z) computes the sigmoid of z.

    from scipy.special import expit 
    import numpy as np

    # You need to return the following variables correctly 
    g = np.zeros(z.shape)

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).


    # g = 1/(1 + np.exp(-z))
    g = expit(z)

    return g

# =============================================================