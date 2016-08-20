import numpy as np

def featureNormalize(X):
    #FEATURENORMALIZE Normalizes the features in X 
    #   FEATURENORMALIZE(X) returns a normalized version of X where
    #   the mean value of each feature is 0 and the standard deviation
    #   is 1. This is often a good preprocessing step to do when
    #   working with learning algorithms.

    mu = np.mean(X, axis=0)
    X_norm = X - mu

    # note that a difference here with the matlab/octave way of handling
    # stddev produces different results further down the pipeline
    # see:
    #   http://stackoverflow.com/q/27600207/583834
    #   https://www.gnu.org/software/octave/doc/v4.0.3/Descriptive-Statistics.html#XREFstd
    # python's np.std() outputs:
    #   [ 1.16126017  1.01312201]
    # octave's std() outputs:
    #   [1.17304991480488,  1.02340777859473]
    # code below uses python np.std(..., ddof=1) following
    #   http://stackoverflow.com/a/27600240/583834
    sigma = np.std(X_norm, axis=0, ddof=1)

    X_norm = X_norm/sigma

    return X_norm, mu, sigma

    # ============================================================

