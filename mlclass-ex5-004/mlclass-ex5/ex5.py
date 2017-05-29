#!/usr/bin/env python

# python adaptation of solved ex5.m
# 

# Regularized Linear Regression and Bias vs. Variance
#
# depends on 
#
#     linearRegCostFunction.py
#     learningCurve.py
#     validationCurve.py
#     trainLinearReg.py
#     polyFeatures.py
#     featureNormalize.py
#     plotFit.py
#

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import linearRegCostFunction as lrcf
import trainLinearReg as tlr
import learningCurve as lc
import polyFeatures as pf
import featureNormalize as fn
import plotFit as pfit
import validationCurve as vc

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...')

mat = scipy.io.loadmat('ex5data1.mat')

X = mat["X"]
y = mat["y"]
Xval = mat["Xval"]
yval = mat["yval"]
Xtest = mat["Xtest"]
ytest = mat["ytest"]

m = X.shape[0]


# Plot training data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show(block=False) 

raw_input('Program paused. Press enter to continue.\n')


## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.array([[1] , [1]])
X_padded = np.column_stack((np.ones((m,1)), X))
J = lrcf.linearRegCostFunction(X_padded, y, theta, 1)

print('Cost at theta = [1 ; 1]: {:f}\n(this value should be about 303.993192)\n'.format(J))

raw_input('Program paused. Press enter to continue.\n')

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

theta = np.array([[1] , [1]])
J, grad = lrcf.linearRegCostFunction(X_padded, y, theta, 1, True)

print('Gradient at theta = [1 ; 1]:  [{:f}; {:f}] \n(this value should be about [-15.303016; 598.250744])'.format(grad[0], grad[1]))

raw_input('Program paused. Press enter to continue.\n')

## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0
lambda_val = 0
theta = tlr.trainLinearReg(X_padded, y, lambda_val)


# resets plot 
plt.close()

#  Plot fit over the data
plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.hold(True)
plt.plot(X, np.dot(np.column_stack((np.ones((m,1)), X)), theta), '--', linewidth=2)
plt.show(block=False)
plt.hold(False)

raw_input('Program paused. Press enter to continue.\n')


## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
#

lambda_val = 0
error_train, error_val = lc.learningCurve(np.column_stack((np.ones((m,1)), X)), y, np.column_stack((np.ones((Xval.shape[0], 1)), Xval)), yval, lambda_val)


# resets plot 
plt.close()

p1, p2 = plt.plot(range(m), error_train, range(m), error_val)
plt.title('Learning curve for linear regression')
plt.legend((p1, p2), ('Train', 'Cross Validation'), numpoints=1, handlelength=0.5)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.show(block=False)
plt.axis([0, 13, 0, 150])

print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in xrange(m):
    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))

raw_input('Program paused. Press enter to continue.\n')

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8;

# Map X onto Polynomial Features and Normalize
X_poly = pf.polyFeatures(X, p)
X_poly, mu, sigma = fn.featureNormalize(X_poly)  # Normalize
X_poly = np.column_stack((np.ones((m,1)), X_poly)) # Add Ones

# # Map X_poly_test and normalize (using mu and sigma)
X_poly_test = pf.polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
X_poly_test = np.column_stack((np.ones((X_poly_test.shape[0],1)), X_poly_test)) # Add Ones

# # Map X_poly_val and normalize (using mu and sigma)
X_poly_val = pf.polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.column_stack((np.ones((X_poly_val.shape[0],1)), X_poly_val)) # Add Ones

print('Normalized Training Example 1:')
print('  {:s}  '.format(X_poly[1, :]))

raw_input('Program paused. Press enter to continue.\n')



## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

lambda_val = 1;
theta = tlr.trainLinearReg(X_poly, y, lambda_val)

# Plot training data and fit
# resets plot 
plt.close()
plt.figure(1)

plt.plot(X, y, 'rx', markersize=10, linewidth=1.5)
pfit.plotFit(min(X), max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)') 
plt.ylabel('Water flowing out of the dam (y)')
plt.title ('Polynomial Regression Fit (lambda = {:f})'.format(lambda_val))
plt.show(block=False)

plt.figure(2)
error_train, error_val = lc.learningCurve(X_poly, y, X_poly_val, yval, lambda_val)
p1, p2 = plt.plot(range(1,m+1), error_train, range(1,m+1), error_val)

plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(lambda_val))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 50])
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.show(block=False)

print('Polynomial Regression (lambda = {:f})\n\n'.format(lambda_val))
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in xrange(m):
    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, float(error_train[i]), float(error_val[i])))

raw_input('Program paused. Press enter to continue.\n')

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

lambda_vec, error_train, error_val = vc.validationCurve(X_poly, y, X_poly_val, yval)

plt.close('all')
p1, p2 = plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.xlabel('lambda')
plt.ylabel('Error')
plt.axis([0, 10, 0, 20])
plt.show(block=False)

print('lambda\t\tTrain Error\tValidation Error\n')
for i in xrange(len(lambda_vec)):
	print(' {:f}\t{:s}\t{:s}\n'.format(lambda_vec[i], error_train[i], error_val[i]))

raw_input('Program paused. Press enter to continue.\n')


## =========== Part 9: Computing test set error on the best lambda found =============
#

# best lambda value from previous step
lambda_val = 3;

# note that we're using X_poly - polynomial linear regression with polynomial features
theta = tlr.trainLinearReg(X_poly, y, lambda_val)

# because we're using X_poly, we also have to use X_poly_test with polynomial features
error_test = lrcf.linearRegCostFunction(X_poly_test, ytest, theta, 0)
print('Test set error: {:f}\n'.format(error_test)) # expected 3.859

raw_input('Program paused. Press enter to continue.\n')


## =========== Part 10: Plot learning curves with randomly selected examples =============
#

# lambda_val value for this step
lambda_val = 0.01

# number of iterations
times = 50

# initialize error matrices
error_train_rand = np.zeros((m, times))
error_val_rand   = np.zeros((m, times))

for i in xrange(1,m+1):

    for k in xrange(times):

        # choose i random training examples
        rand_sample_train = np.random.permutation(X_poly.shape[0])
        rand_sample_train = rand_sample_train[:i]

        # choose i random cross validation examples
        rand_sample_val   = np.random.permutation(X_poly_val.shape[0])
        rand_sample_val   = rand_sample_val[:i]

        # define training and cross validation sets for this loop
        X_poly_train_rand = X_poly[rand_sample_train,:]
        y_train_rand      = y[rand_sample_train]
        X_poly_val_rand   = X_poly_val[rand_sample_val,:]
        yval_rand         = yval[rand_sample_val]

        print X_poly_train_rand.shape 
        print y_train_rand.shape      
        print X_poly_val_rand.shape   
        print yval_rand.shape                 

        # note that we're using X_poly_train_rand and y_train_rand in training
        theta = tlr.trainLinearReg(X_poly_train_rand, y_train_rand, lambda_val)
            
        # we use X_poly_train_rand, y_train_rand, X_poly_train_rand, X_poly_val_rand
        error_train_rand[i-1,k] = lrcf.linearRegCostFunction(X_poly_train_rand, y_train_rand, theta, 0)
        error_val_rand[i-1,k]   = lrcf.linearRegCostFunction(X_poly_val_rand,   yval_rand,    theta, 0)


error_train = np.mean(error_train_rand, axis=1)
error_val   = np.mean(error_val_rand, axis=1)

# resets plot 
plt.close()

p1, p2 = plt.plot(range(m), error_train, range(m), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(lambda_val))
plt.legend((p1, p2), ('Train', 'Cross Validation'))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show(block=False)


print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in xrange(m):
    print('  \t{:d}\t\t{:f}\t{:f}\n'.format(i+1, error_train[i], error_val[i]))

raw_input('Program paused. Press enter to continue.\n')
