def oneVsAll(X, y, num_labels, lambda_reg):
#ONEVSALL trains multiple logistic regression classifiers and returns all
#the classifiers in a matrix all_theta, where the i-th row of all_theta 
#corresponds to the classifier for label i
#   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
#   logisitc regression classifiers and returns each of these classifiers
#   in a matrix all_theta, where the i-th row of all_theta corresponds 
#   to the classifier for label i

    import numpy as np
    import lrCostFunction as lrcf
    from scipy.optimize import fmin_cg, minimize

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly 
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m,1)), X))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda. 
    #

    for c in range(num_labels):

    	# initial theta for c/class
    	initial_theta = np.zeros((n + 1, 1))

        print("Training {:d} out of {:d} categories...".format(c+1, num_labels))

        # Set options for fmin_cg
        myargs = (X, (y%10==c).astype(int), lambda_reg, True)

    	# use fmin_cg to train oneVsAll classifier
        # theta = fmin_cg(lrcf.lrCostFunction, x0=initial_theta, args=myargs, maxiter=50)
    	theta = minimize(lrcf.lrCostFunction, x0=initial_theta, args=myargs, method="Newton-CG", jac=True)

    	# assign row of all_theta corresponding to current c/class
    	all_theta[c,:] = theta["x"]

    # =========================================================================

    return all_theta
