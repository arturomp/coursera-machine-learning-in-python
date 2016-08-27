import numpy as np
import cofiCostFunc as ccf
import computeNumericalGradient as cng

def checkCostFunction(lambda_var=0):
    #CHECKCOSTFUNCTION Creates a collaborative filering problem 
    #to check your cost function and gradients
    #   CHECKCOSTFUNCTION(lambda_var) Creates a collaborative filering problem 
    #   to check your cost function and gradients, it will output the 
    #   analytical gradients produced by your code and the numerical gradients 
    #   (computed using computeNumericalGradient). These two gradient 
    #   computations should result in very similar values.

    # Set lambda_var
    # if not lambda_var or not 'lambda_var' in locals():
    #     lambda_var = 0

    ## Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    ## Run Gradient Checking
    X = np.random.randn(X_t.shape[0], X_t.shape[1])
    Theta = np.random.randn(Theta_t.shape[0], Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

    # Short hand for cost function
    def costFunc(p):
        return ccf.cofiCostFunc(p, Y, R, num_users, num_movies, num_features, lambda_var)

    numgrad = cng.computeNumericalGradient(costFunc, params)

    cost, grad = ccf.cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_var)


    print(np.column_stack((numgrad, grad)))
    print('The above two columns you get should be very similar.\n' \
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your backpropagation implementation is correct, then \n' \
             'the relative difference will be small (less than 1e-9). ' \
             '\nRelative Difference: {:e}'.format(diff))

