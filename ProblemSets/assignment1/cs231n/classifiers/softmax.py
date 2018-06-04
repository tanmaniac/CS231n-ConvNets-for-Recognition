import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # Compute the softmax loss and its gradient using explicit loops.           #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # Softmax loss: L_i = -log(e^s_y_i/sigma_j(e^s_j))
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    for tr in range(num_train):
        scores = X[tr].dot(W)

        # Shift scores by the maximum value to avoid numeric instability
        scores -= np.max(scores)
        class_exp = np.exp(scores)

        # Compute Softmax for whole row
        row_sum = np.sum(class_exp)
        row_softmax = class_exp / row_sum
        S_i = row_softmax[y[tr]]

        # Update loss
        loss += -np.log(S_i)

        # Compute gradient
        kronecker_delta = np.zeros(scores.shape)
        kronecker_delta[y[tr]] = 1
        dW += X[tr].reshape(-1, 1) * (row_softmax -
                                      kronecker_delta).reshape(1, -1)

    # Take the average of the computed loss over all the training data
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # Compute the softmax loss and its gradient using no explicit loops.        #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # Compute scores
    scores = X.dot(W)
    # Shift scores to avoid numeric instability
    row_sums = np.max(scores, axis=1)
    scores = (scores.T - row_sums.T).T

    # Compute softmax
    exps = np.exp(scores)
    softmaxes = (exps.T / np.sum(exps, axis=1).T).T
    S_i = softmaxes[range(num_train), y]

    # Compute loss
    loss = np.sum(-np.log(S_i)) / num_train + reg * np.sum(W * W)

    # Compute gradient
    kronecker_delta = np.zeros(scores.shape)
    kronecker_delta[range(num_train), y] = 1
    diffs = softmaxes - kronecker_delta
    dW = X.T.dot(diffs) / num_train + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
