import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        num_below_margin = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # Compute gradient for j != y_i
                dW[:, j] += X[i]
                num_below_margin += 1
        # Gradient where j == y_i
        dW[:, y[i]] -= num_below_margin * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = X.dot(W)
    # Row vector of scores for the correct class
    correct_scores = scores[range(num_train), y].reshape(-1)
    # Compute margins. Insane transposes because broadcasting only works across rows
    margins = np.maximum(0, (scores.T - correct_scores.T).T + 1)
    # Set indices corresponding to the correct scores to zero
    margins[range(num_train), y] = 0

    loss = np.sum(margins) / num_train + reg * np.sum(W * W)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    class_mask = np.zeros(margins.shape)
    class_mask[margins != 0] = 1    # Nx10 mask; 1 where score > margin and 0 otherwise

    num_below_margin = np.sum(class_mask, axis=1)   # Nx1; number of class predictions > 0
    
    # Set the number of predictions below the margin for each correct class
    class_mask[range(num_train), y] = -num_below_margin

    # Accumulate sums and add regularization
    dW = X.T.dot(class_mask) / num_train + reg * W
    
    #print(dW.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
