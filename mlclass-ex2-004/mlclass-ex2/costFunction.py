def costFunction(theta, X, y, return_grad=False):
#COSTFUNCTION Compute cost and gradient for logistic regression
#   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
#   parameter for logistic regression and the gradient of the cost
#   w.r.t. to the parameters.

    import numpy as np 
    from sigmoid import sigmoid

    # Initialize some useful values
    m = len(y) # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta
    #

    # given the following dimensions:
    # theta.shape = (n+1,1)
    # X.shape     = (m,n+1)
    # the equation's 
    #	theta' times X
    # becomes
    # 	np.dot(X,theta)
    # to obtain a (m,1) vector
    # given that
    #   y.shape     = (m,)
    # we transpose the (m,1) shaped 
    #   np.log( sigmoid( np.dot(X,theta) ) )        , as well as
    #   np.log( 1 - sigmoid( np.dot(X,theta) ) )
    # to obtain (1,m) vectors to be mutually added, 
    # and whose elements are summed to form a scalar 
    one = y * np.transpose(np.log( sigmoid( np.dot(X,theta) ) ))
    two = (1-y) * np.transpose(np.log( 1 - sigmoid( np.dot(X,theta) ) ))
    J = -(1./m)*(one+two).sum()

    # here we need n+1 gradients. 
    # note that 
    #   y.shape                          = (m,)
    #   sigmoid( np.dot(X,theta) ).shape = (m, 1)
    # so we transpose the latter, subtract y, obtaining a vector of (1, m)
    # we multiply such vector by X, whose dimension is 
    #   X.shape = (m, n+1), 
    # and we obtain a (1, n+1) vector, which we also transpose
    # this last vectorized multiplication takes care of the sum
    grad = (1./m) * np.dot(sigmoid( np.dot(X,theta) ).T - y, X).T

    if return_grad == True:
        return J, np.transpose(grad)
    elif return_grad == False:
        return J # for use in fmin/fmin_bfgs optimization function

# =============================================================
