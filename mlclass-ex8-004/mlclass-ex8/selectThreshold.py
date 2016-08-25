import numpy as np

def selectThreshold(yval, pval):
    #SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
    #outliers
    #   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
    #   threshold to use for selecting outliers based on the results from a
    #   validation set (pval) and the ground truth (yval).
    #

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), stepsize):
        
        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #               
        # Note: You can use predictions = (pval < epsilon) to get a binary vector
        #       of 0's and 1's of the outlier predictions

        # make anomaly predictions based on which values of pval are less than epsilon
        #   i.e. which examples from the cross-validation set have a very low p(x)
        cvPredictions = pval < epsilon

        # calculate F1 score, starting by calculating true positives(tp)
        # false positives (fp) and false negatives (fn)

        # true positives is the intersection between 
        #   our positive predictions (cvPredictions==1) and positive ground truth values (yval==1)
        tp = np.sum(np.logical_and((cvPredictions==1), (yval==1)).astype(float))

        # false positives are the ones we predicted to be true (cvPredictions==1) but weren't (yval==0)
        fp = np.sum(np.logical_and((cvPredictions==1), (yval==0)).astype(float))
        
        # false negatives are the ones we said were false (cvPredictions==0) but which were true (yval==1)
        fn = np.sum(np.logical_and((cvPredictions==0), (yval==1)).astype(float))

        # compute precision, recall and F1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1 = (2*precision*recall)/(precision+recall)

        # =============================================================

        if F1 > bestF1:
           bestF1 = F1
           bestEpsilon = epsilon

    return bestEpsilon, bestF1