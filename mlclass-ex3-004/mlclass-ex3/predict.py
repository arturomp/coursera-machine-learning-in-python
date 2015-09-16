def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

    import sigmoid as s
    import numpy as np

    # turns 1D X array into 2D
    if X.ndim == 1:
        X = np.reshape(X, (-1,X.shape[0]))

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros((m,1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a 
    #               vector containing labels between 1 to num_labels.
    #

    # add column of ones as bias unit from input layer to second layer
    X = np.column_stack((np.ones((m,1)), X)) # = a1

    # calculate second layer as sigmoid( z2 ) where z2 = Theta1 * a1
    a2 = s.sigmoid( np.dot(X,Theta1.T) )

    # add column of ones as bias unit from second layer to third layer
    a2 = np.column_stack((np.ones((a2.shape[0],1)), a2))

    # calculate third layer as sigmoid ( z3 ) where z3 = Theta2 * a2
    a3 = s.sigmoid( np.dot(a2,Theta2.T) )

    # get indices as in predictOneVsAll
    p = np.argmax(a3, axis=1)

    # =========================================================================

    return p + 1 # offsets python's zero notation
