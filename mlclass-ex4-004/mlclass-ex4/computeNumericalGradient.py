import numpy as np
import nnCostFunction as nncf

def computeNumericalGradient(theta, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lambda_reg):
    #COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
    #and gives us a numerical estimate of the gradient.
    #   numgrad = COMPUTENUMERICALGRADIENT(nn_params, input_layer_size, 
    #                                      hidden_layer_size,  num_labels, 
    #                                      X, y, lambda_reg) 
    #   computes the numerical gradient of the cost function around theta.

    # Notes: The following code implements numerical gradient checking, and 
    #        returns the numerical gradient.It sets numgrad(i) to (a numerical 
    #        approximation of) the partial derivative of J with respect to the 
    #        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
    #        be the (approximately) the partial derivative of J with respect 
    #        to theta(i).)
    #                

    numgrad = np.zeros( theta.shape )
    perturb = np.zeros( theta.shape )
    e = 1e-4

    for p in xrange(theta.size):
        # Set perturbation vector
        perturb.reshape(perturb.size, order="F")[p] = e
        loss1, _ = nncf.nnCostFunction(theta - perturb, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lambda_reg)
        loss2, _ = nncf.nnCostFunction(theta + perturb, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lambda_reg)
        # Compute Numerical Gradient
        numgrad.reshape(numgrad.size, order="F")[p] = (loss2 - loss1) / (2*e)
        perturb.reshape(perturb.size, order="F")[p] = 0

    return numgrad