import numpy as np
import sigmoid as s

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
	num_labels, X, y, lambda_reg):
    #NNCOSTFUNCTION Implements the neural network cost function for a two layer
    #neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices. 
    # 
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.


    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], \
                     (hidden_layer_size, input_layer_size + 1), order='F')

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], \
                     (num_labels, hidden_layer_size + 1), order='F')

    # Setup some useful variables
    m = len(X)
             
    # # You need to return the following variables correctly 
    J = 0;
    Theta1_grad = np.zeros( ( len(Theta1),len(Theta1[0]) ) );
    Theta2_grad = np.zeros( ( len(Theta2),len(Theta2[0]) ) );

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a 
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the 
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #

    # add column of ones as bias unit from input layer to second layer
    X = np.column_stack((np.ones((m,1)), X)) # = a1

    # calculate second layer as sigmoid( z2 ) where z2 = Theta1 * a1
    a2 = s.sigmoid( np.dot(X,Theta1.T) )

    # add column of ones as bias unit from second layer to third layer
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))

    # calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
    a3 = s.sigmoid( np.dot(a2,Theta2.T) )

    #%% COST FUNCTION CALCULATION

    #% NONREGULARIZED COST FUNCTION

    # recode labels as vectors containing only values 0 or 1
    labels = y
    # set y to be matrix of size m x k
    y = np.zeros((m,num_labels));
    # for every label, convert it into vector of 0s and a 1 in the appropriate position
    for i in xrange(m):
    	y[i, labels[i]-1] = 1

    # at this point, both a3 and y are m x k matrices, where m is the number of inputs
    # and k is the number of hypotheses. Given that the cost function is a sum
    # over m and k, loop over m and in each loop, sum over k by doing a sum over the row

    cost = 0
    for i in xrange(m):
    	cost += np.sum( y[i] * np.log( a3[i] ) + (1 - y[i]) * np.log( 1 - a3[i] ) )

    J = -(1.0/m)*cost

    #% REGULARIZED COST FUNCTION
    # note that Theta1[:,1:] is necessary given that the first column corresponds to transitions
    # from the bias terms, and we are not regularizing those parameters. Thus, we get rid
    # of the first column.

    sumOfTheta1 = np.sum(np.sum(Theta1[:,1:]**2))
    sumOfTheta2 = np.sum(np.sum(Theta2[:,1:]**2))

    J = J + ( (lambda_reg/(2.0*m))*(sumOfTheta1+sumOfTheta2) )

    return J

    # #%% BACKPROPAGATION

    # bigDelta1 = 0;
    # bigDelta2 = 0;

    # # for each training example
    # for t = 1:m,


    # 	## step 1: perform forward pass
    # 	# set lowercase x to the t-th row of X
    # 	x = X(t,:);
    # 	# note that uppercase X already included column of ones 
    # 	# as bias unit from input layer to second layer, so no need to add it

    # 	# calculate second layer as sigmoid( z2 ) where z2 = Theta1 * a1
    # 	a2 = sigmoid( x * Theta1' );

    # 	# add column of ones as bias unit from second layer to third layer
    # 	a2 = [ones(size(a2,1),1) a2];

    # 	# calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
    # 	a3 = sigmoid( a2 * Theta2' );



    # 	## step 2: for each output unit k in layer 3, set delta_{k}^{(3)}
    # 	delta3 = zeros(1, num_labels);

    # 	# see handout for more details, but y_k indicates whether  
    # 	# the current training example belongs to class k (y_k = 1), 
    # 	# or if it belongs to a different class (y_k = 1)
    # 	for k = 1:num_labels,
    # 		y_k = y(t,k) == 1;
    # 		delta3(k) = a3(k) - y_k;
    # 	end;

    # 	## step 3: for the hidden layer l=2, set delta2 = Theta2' * delta3 .* sigmoidGradient(z2)
    # 	# note that we're skipping delta2_0 (=gradients of bias units, which we don't use here)
    # 	# by doing (Theta2(:,2:end))' instead of Theta2'
    # 	delta2 = ((Theta2(:,2:end))' * delta3')' .* sigmoidGradient( x * Theta1' );

    # 	## step 4: accumulate gradient from this example
    # 	# accumulation
    # 	bigDelta1 = bigDelta1 + delta2'*x;
    # 	bigDelta2 = bigDelta2 + delta3'*a2;


    # # step 5: obtain gradient for neural net cost function by dividing the accumulated gradients by m
    # Theta1_grad = bigDelta1 / m;
    # Theta2_grad = bigDelta2 / m;

    # #% REGULARIZATION FOR GRADIENT
    # # only regularize for j >= 1, so skip the first column
    # Theta1_grad_unregularized = Theta1_grad;
    # Theta2_grad_unregularized = Theta2_grad;
    # Theta1_grad = Theta1_grad + (lambda_reg/m)*Theta1;
    # Theta2_grad = Theta2_grad + (lambda_reg/m)*Theta2;
    # Theta1_grad(:,1) = Theta1_grad_unregularized(:,1);
    # Theta2_grad(:,1) = Theta2_grad_unregularized(:,1);

    # # -------------------------------------------------------------

    # # =========================================================================

    # # Unroll gradients
    # grad = [Theta1_grad(:) ; Theta2_grad(:)];

