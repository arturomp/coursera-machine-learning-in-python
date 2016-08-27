import numpy as np

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_var):
    #COFICOSTFUNC Collaborative filtering cost function
    #   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
    #   num_features, lambda) returns the cost and gradient for the
    #   collaborative filtering problem.
    #

    # Unfold the U and W matrices from params
    X = np.reshape(params[:num_movies*num_features], (num_movies, num_features), order='F')
    Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features), order='F')

    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the 
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the 
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the 
    #                     partial derivatives w.r.t. to each element of Theta
    #

    ### COST FUNCTION, NO REGULARIZATION

    # X * Theta performed according to low rank matrix vectorization
    squared_error = np.power(np.dot(X,Theta.T) - Y,2)

    # for cost function, sum only i,j for which R(i,j)=1
    J = (1/2.) * np.sum(squared_error * R)

    ### GRADIENTS, NO REGULARIZATION

    # X_grad is of dimensions n_m x n, where n_m = 1682 and n = 100 
    #  ( (X * Theta') - Y ) is n_m x n_u
    #  (( (X * Theta') - Y ) .* R) is still n_m x n_u 
    #  Theta is n_u x n
    #  thus X_grad = (( (X * Theta') - Y ) .* R) * Theta is n_m x n
    # NOTE where filtering through R is applied: 
    # 	at ( (X * Theta') - Y ) - NOT after multiplying by Theta 
    # 	that means that for purposes of the gradient, we're only interested
    # 	in the errors/differences in i, j for which R(i,j)=1
    # NOTE also that even though we do a sum, we only do it over users,
    # 	so we still get a matrix
    X_grad = np.dot(( np.dot(X, Theta.T) - Y ) * R, Theta)


    # Theta_grad is of dimensions n_u x n, where n_u = 943 and n = 100 
    #  ( (X * Theta') - Y ) is n_m x n_u 
    #  (( (X * Theta') - Y ) .* R) is still n_m x n_u 
    #  X is n_m x n
    #  thus Theta_grad = (( (X * Theta') - Y ) .* R)' * X is n_u x n 
    # NOTE where filtering through R is applied,
    # 	at ( (X * Theta') - Y ) - NOT after multiplying by X
    # 	that means that for purposes of the gradient, we're only interested
    # 	in the errors/differences in i, j for which R(i,j)=1
    # NOTE also that even though we do a sum, we only do it over movies,
    # 	so we still get a matrix
    Theta_grad = np.dot((( np.dot(X, Theta.T) - Y ) * R).T, X)

    ### COST FUNCTION WITH REGULARIZATION
    # only add regularized cost to J now
    J = J + (lambda_var/2.)*( np.sum( np.power(Theta, 2) ) + np.sum( np.power(X, 2) ) )

    ### GRADIENTS WITH REGULARIZATION
    # only add regularization terms
    X_grad = X_grad + lambda_var*X
    Theta_grad = Theta_grad + lambda_var*Theta

    # =============================================================

    grad = np.concatenate((X_grad.reshape(X_grad.size, order='F'), Theta_grad.reshape(Theta_grad.size, order='F')))

    return J, grad