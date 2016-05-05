#!/usr/bin/env python

# python adaptation of solved ex4.m
# 
# Neural network learning
# 
# depends on 
#
#     displayData.py
#     sigmoidGradient.py
#     randInitializeWeights.py
#     nnCostFunction.py
#

import scipy.io
import numpy as np
import displayData as dd
import nnCostFunction as nncf

## Setup the parameters you will use for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('ex4data1.mat')

X = mat["X"]
y = mat["y"]

m = X.shape[0]

# crucial step in getting good performance!
# changes the dimension from (m,1) to (m,)
# otherwise the minimization isn't very effective...
y=y.flatten() 

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[:100],:]

dd.displayData(sel)

raw_input('Program paused. Press enter to continue.\n')

## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
mat = scipy.io.loadmat('ex4weights.mat')
Theta1 = mat["Theta1"]
Theta2 = mat["Theta2"]

# Unroll parameters 
# ndarray.flatten() always creates copy (http://stackoverflow.com/a/28930580/583834)
# ndarray.ravel() requires transpose to have matlab unrolling order (http://stackoverflow.com/a/15988852/583834)
# np.append() always makes a copy (http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.append.html)
nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('Feedforward Using Neural Network ...')

# # Weight regularization parameter (we set this to 0 here).
lambda_reg = 0;

J = nncf.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lambda_reg)


print('Training Set Accuracy: {:f}\n(this value should be about 0.287629)'.format(J))

raw_input('Program paused. Press enter to continue.\n')

# ## =============== Part 4: Implement Regularization ===============
# #  Once your cost function implementation is correct, you should now
# #  continue to implement the regularization with the cost.
# #

# print('\nChecking Cost Function (w/ Regularization) ... \n')

# # Weight regularization parameter (we set this to 1 here).
# lambda_reg = 1;

# J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
#                    num_labels, X, y, lambda_reg);

# print(['Cost at parameters (loaded from ex4weights): %f '...
#          '\n(this value should be about 0.383770)\n'], J);

# print('Program paused. Press enter to continue.\n');
# pause;


# ## ================ Part 5: Sigmoid Gradient  ================
# #  Before you start implementing the neural network, you will first
# #  implement the gradient for the sigmoid function. You should complete the
# #  code in the sigmoidGradient.m file.
# #

# print('\nEvaluating sigmoid gradient...\n')

# g = sigmoidGradient([1 -0.5 0 0.5 1]);
# print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
# print('%f ', g);
# print('\n\n');

# print('Program paused. Press enter to continue.\n');
# pause;


# ## ================ Part 6: Initializing Pameters ================
# #  In this part of the exercise, you will be starting to implment a two
# #  layer neural network that classifies digits. You will start by
# #  implementing a function to initialize the weights of the neural network
# #  (randInitializeWeights.m)

# print('\nInitializing Neural Network Parameters ...\n')

# initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
# initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

# # Unroll parameters
# initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


# ## =============== Part 7: Implement Backpropagation ===============
# #  Once your cost matches up with ours, you should proceed to implement the
# #  backpropagation algorithm for the neural network. You should add to the
# #  code you've written in nnCostFunction.m to return the partial
# #  derivatives of the parameters.
# #
# print('\nChecking Backpropagation... \n');

# #  Check gradients by running checkNNGradients
# checkNNGradients;

# print('\nProgram paused. Press enter to continue.\n');
# pause;


# ## =============== Part 8: Implement Regularization ===============
# #  Once your backpropagation implementation is correct, you should now
# #  continue to implement the regularization with the cost and gradient.
# #

# print('\nChecking Backpropagation (w/ Regularization) ... \n')

# #  Check gradients by running checkNNGradients
# lambda_reg = 3;
# checkNNGradients(lambda_reg);

# # Also output the costFunction debugging values
# debug_J  = nnCostFunction(nn_params, input_layer_size, ...
#                           hidden_layer_size, num_labels, X, y, lambda_reg);

# print(['\n\nCost at (fixed) debugging parameters (w/ lambda_reg = 10): %f ' ...
#          '\n(this value should be about 0.576051)\n\n'], debug_J);

# print('Program paused. Press enter to continue.\n');
# pause;


# ## =================== Part 8: Training NN ===================
# #  You have now implemented all the code necessary to train a neural 
# #  network. To train your neural network, we will now use "fmincg", which
# #  is a function which works similarly to "fminunc". Recall that these
# #  advanced optimizers are able to train our cost functions efficiently as
# #  long as we provide them with the gradient computations.
# #
# print('\nTraining Neural Network... \n')

# #  After you have completed the assignment, change the MaxIter to a larger
# #  value to see how more training helps.
# options = optimset('MaxIter', 200);

# #  You should also try different values of lambda_reg
# lambda_reg = 0.1;

# # Create "short hand" for the cost function to be minimized
# costFunction = @(p) nnCostFunction(p, ...
#                                    input_layer_size, ...
#                                    hidden_layer_size, ...
#                                    num_labels, X, y, lambda_reg);

# # Now, costFunction is a function that takes in only one argument (the
# # neural network parameters)
# [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

# # Obtain Theta1 and Theta2 back from nn_params
# Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
#                  hidden_layer_size, (input_layer_size + 1));

# Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
#                  num_labels, (hidden_layer_size + 1));

# print('Program paused. Press enter to continue.\n');
# pause;


# ## ================= Part 9: Visualize Weights =================
# #  You can now "visualize" what the neural network is learning by 
# #  displaying the hidden units to see what features they are capturing in 
# #  the data.

# print('\nVisualizing Neural Network... \n')

# displayData(Theta1(:, 2:end));

# print('\nProgram paused. Press enter to continue.\n');
# pause;

# ## ================= Part 10: Implement Predict =================
# #  After training the neural network, we would like to use it to predict
# #  the labels. You will now implement the "predict" function to use the
# #  neural network to predict the labels of the training set. This lets
# #  you compute the training set accuracy.

# pred = predict(Theta1, Theta2, X);

# print('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


