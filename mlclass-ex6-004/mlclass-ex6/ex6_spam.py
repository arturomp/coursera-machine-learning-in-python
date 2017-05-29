#!/usr/bin/env python

# python adaptation of solved ex6_spam.m
#

#  Spam Classification with SVMs
#
#  depends on
#
#     readFile.py
#     processEmail.py
#     emailFeatures.py
#     svmTrain.py
#     getVocabList.py
#     NLTK package (for Porter stemmer)
#

import scipy.io
import numpy as np
import readFile as rf
import processEmail as pe
import emailFeatures as ef
import svmTrain as svmt
import getVocabList as gvl


## ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

print('Preprocessing sample email (emailSample1.txt)')

# Extract Features
file_contents = rf.readFile('emailSample1.txt')
word_indices  = pe.processEmail(file_contents)

# Print Stats
print('Word Indices: ')
print(' {:s}'.format(word_indices))
print('\n\n')

raw_input('Program paused. Press enter to continue.')

## ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n. 
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

print('Extracting features from sample email (emailSample1.txt)')

# Extract Features
file_contents = rf.readFile('emailSample1.txt')
word_indices  = pe.processEmail(file_contents)
features      = ef.emailFeatures(word_indices)

# Print Stats
print('Length of feature vector: {:d}'.format( len(features) ) )
print('Number of non-zero entries: {:d}'.format( np.sum(features > 0) ) )

raw_input('Program paused. Press enter to continue.')

## =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
mat = scipy.io.loadmat('spamTrain.mat')
X = mat["X"]
y = mat["y"]

y = y.flatten()

print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...')

C = 0.1
model = svmt.svmTrain(X, y, C, "linear")

p = model.predict(X)

raw_input('Training Accuracy: {:f}'.format( np.mean((p == y).astype(int)) * 100 ))

## =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
mat = scipy.io.loadmat('spamTest.mat')
Xtest = mat["Xtest"]
ytest = mat["ytest"]

ytest = ytest.flatten()

print('Evaluating the trained Linear SVM on a test set ...')

p = model.predict(Xtest)

raw_input('Test Accuracy: {:f}'.format( np.mean((p == ytest).astype(int)) * 100 ))


## ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.
#

# Sort the weights and obtain the vocabulary list
w = model.coef_[0]

# from http://stackoverflow.com/a/16486305/583834
# reverse sorting by index
indices = w.argsort()[::-1][:15]
vocabList = sorted(gvl.getVocabList().keys())

print('\nTop predictors of spam: \n');
for idx in indices: 
    print( ' {:s} ({:f}) '.format( vocabList[idx], float(w[idx]) ) )

raw_input('Program paused. Press enter to continue.')

## =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
#  The following code reads in one of these emails and then uses your 
#  learned SVM classifier to determine whether the email is Spam or 
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
filename = 'spamSample1.txt'

# Read and predict
file_contents = rf.readFile(filename)
word_indices  = pe.processEmail(file_contents)
x             = ef.emailFeatures(word_indices)
p = model.predict(x.flatten())

print('\nProcessed {:s}\n\nSpam Classification: {:s}\n'.format(filename, p))
print('(1 indicates spam, 0 indicates not spam)\n\n')

filename = 'spamSample2.txt'

# Read and predict
file_contents = rf.readFile(filename)
word_indices  = pe.processEmail(file_contents)
x             = ef.emailFeatures(word_indices)
p = model.predict(x.flatten())

print('\nProcessed {:s}\n\nSpam Classification: {:s}\n'.format(filename, p))
print('(1 indicates spam, 0 indicates not spam)\n\n')

filename = 'emailSample1.txt'

# Read and predict
file_contents = rf.readFile(filename)
word_indices  = pe.processEmail(file_contents)
x             = ef.emailFeatures(word_indices)
p = model.predict(x.flatten())

print('\nProcessed {:s}\n\nSpam Classification: {:s}\n'.format(filename, p))
print('(1 indicates spam, 0 indicates not spam)\n\n')

filename = 'emailSample2.txt'

# Read and predict
file_contents = rf.readFile(filename)
word_indices  = pe.processEmail(file_contents)
x             = ef.emailFeatures(word_indices)
p = model.predict(x.flatten())

print('\nProcessed {:s}\n\nSpam Classification: {:s}\n'.format(filename, p))
print('(1 indicates spam, 0 indicates not spam)\n\n')