def plotDecisionBoundary(theta, X, y):
#PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
#the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
#   positive examples and o for the negative examples. X is assumed to be 
#   a either 
#   1) Mx3 matrix, where the first column is an all-ones column for the 
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones

    import matplotlib.pyplot as plt
    import numpy as np

    # Plot Data
    fig = plt.figure()

    # Find Indices of Positive and Negative Examples
    pos = np.where(y==1);
    neg = np.where(y==0);

    # plot! [0] indexing at end necessary for proper legend creation
    p1 = plt.plot(X[pos,0], X[pos,1], marker='+', markersize=9, color='k')[0]
    p2 = plt.plot(X[neg,0], X[neg,1], marker='o', markersize=7, color='y')[0]

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:,1])-2,  max(X[:,1])+2])

        # Calculate the decision boundary line
        plot_y = (-1./theta[2])*(theta[1]*plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        p3 = plt.plot(plot_x, plot_y)
        
        # Legend, specific for the exercise
        plt.legend((p1, p2, p3[0]), ('Admitted', 'Not Admitted', 'Decision Boundary'), numpoints=1, handlelength=0.5)

        plt.axis([30, 100, 30, 100])

        plt.show(block=False)

    plt.hold(False) # prevents further drawing on plot

    return plt
