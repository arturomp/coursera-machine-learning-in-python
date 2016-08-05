import numpy as np
import svmTrain as svmt
import gaussianKernelGramMatrix as gkgm

def dataset3Params(X, y, Xval, yval):
    #EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
    #where you select the optimal (C, sigma) learning parameters to use for SVM
    #with RBF kernel
    #   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
    #   sigma. You should complete this function to return the optimal C and 
    #   sigma based on a cross-validation set.
    #

    # You need to return the following variables correctly.
    sigma = 0.3
    C = 1

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use svmPredict to predict the labels on the cross
    #               validation set. For example, 
    #                   predictions = svmPredict(model, Xval)
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using 
    #        mean(double(predictions ~= yval))
    #

    ### determining best C and sigma

    # need x1 and x2, copied from ex6.py
    x1 = [1, 2, 1] 
    x2 = [0, 4, -1]

    # only uncomment if similar lines are uncommented on svmTrain.py
    # yval = yval.astype("int32")
    # yval[yval==0] = -1

    # vector with all predictions from SVM
    predictionErrors = np.zeros((64,3))
    predictionsCounter = 0

    # iterate over values of sigma and C
    for sigma in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:
        for C in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]:

            print "trying sigma={:.2f}, C={:.2f}".format(sigma, C)
            
            # train model on training corpus with current sigma and C
            model = svmt.svmTrain(X, y, C, "gaussian", sigma=sigma)

            # compute predictions on cross-validation set
            predictions = model.predict(gkgm.gaussianKernelGramMatrix(Xval, X))

            # compute prediction errors on cross-validation set
            predictionErrors[predictionsCounter,0] = np.mean((predictions != yval).astype(int))

            # store corresponding C and sigma
            predictionErrors[predictionsCounter,1] = sigma      
            predictionErrors[predictionsCounter,2] = C      
            
            # move counter up by one
            predictionsCounter = predictionsCounter + 1

    print(predictionErrors)

    # calculate mins of columns with their indexes
    row = predictionErrors.argmin(axis=0)
    m   = np.zeros(row.shape)
    for i in xrange(len(m)):
        m[i] = predictionErrors[row[i]][i]


    # note that row[0] is the index of the min of the first column
    #   and that the first column corresponds to the error, 
    #   so the row at predictionErrors(row(1),:) has best C and sigma
    print(predictionErrors[row[0],1])
    print(predictionErrors[row[0],2])

    # get C and sigma form such row
    sigma = predictionErrors[row[0],1]
    C     = predictionErrors[row[0],2]


    return C, sigma
