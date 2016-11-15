# - *- coding: utf- 8 - *-
import theano
import theano.tensor as T
import numpy as np
from utils import *

def mean_squared_error_loss(y_pred, y_actual, **kwargs):
    return(T.mean(T.square(y_pred - y_actual), axis=-1))

def mean_absolute_error_loss(y_pred, y_actual, **kwargs):
    return(T.mean(T.abs_(y_pred - y_actual), axis=-1))

def mean_absolute_percentage_error_loss(y_pred, y_actual, **kwargs):
    print("Use mean absolute percentage error. Ensure no outputs are exactly 0.")
    diff = T.abs_( (y_actual - y_pred) / T.clip(T.abs_(y_actual), epsilon, np.inf))
    return(100. * T.mean(diff, axis=-1))
    # Error is not defined
    # at y_actual = 0 so we substitute with some small value epsilon

def mean_squared_logarithmic_loss(y_pred, y_actual, **kwargs):
    # Intended for case where only positive values are outputs. It penalizes under-predicted values more than
    # overpredicted values
    print("Using mean squared loss. Ensure all outputs (pred and actual) are >= 0")
    log_pred = T.log( T.clip(y_pred, epsilon, np.inf) + 1. )
    log_actual = T.log( T.clip(y_actual, epsilon, np.inf) + 1. )
    return( T.mean(T.square(log_pred - log_actual), axis=-1) )

def squared_hinge_loss(y_pred, y_actual, **kwargs):
    # Used for training classifiers. Assumes the true labels are +1 and -1 (and NOT 0)
    print("Using squared hinge loss. Ensure labels are +1, -1 and NOT 0.")
    return T.mean(T.square(T.maximum(1. - y_actual * y_pred, 0.)), axis=-1)
    # To minimize this loss, both y_tue and y_pred must be of the same sign and their product must exceed or equal 1.
    # Note that y_actual is the +1 or -1 class label and y_pred is the real value probability of that class predicted by
    # the model.

def hinge_loss(y_pred, y_actual, **kwargs):
    # Used for training classifiers. Assumes the true labels are +1 and -1 (and NOT 0)
    print("Using hinge loss. Ensure labels are +1 -1 and not 0.")
    return T.mean(T.maximum(1. - y_actual * y_pred, 0.), axis=-1)
    # To minimize this loss, both y_tue and y_pred must be of the same sign and their product must exceed or equal 1.
    # Note that y_actual is the +1 or -1 class label and y_pred is the real value probability of that class predicted by
    # the model.

def categorical_crossentropy_loss(y_pred, y_actual, from_logits=False, **kwargs):
    """
    Expects a binary class matrix instead of a vector of scalar classes. This evaluates the cross-entropy between the
    true distribution y_actual and y_pred defined as:
        H(y_actual, y_pred) = - sum_{label} ( y_actual(label) * log(y_pred(label)) )
    In case of classification with y_actual being a 1-hot class indicator, cross-entropy reduces to the sum of negative
    log of predicted probability of the correct label.
    """
    print("Using categorical cross-entropy. Ensure y_actual is a matrix with each row  have appropriate class indicator"
          "variables for the corresponding input sample. Current implementation of cross entropy averages "
          "across samples. This should make the learning rate robust to the size of the dataset.")
    assert y_actual.ndim == 2 and y_pred.ndim == 2, "Supplied gold and pred distributions for CCE are not appropriate!"
    if from_logits:
        # Output of a layer with un-normalized, possibility negative scores. This is simply a precaution to ensure
        # compatibility with categorical cross-entropy.
        y_pred = T.nnet.softmax(y_pred)
    else:
        # scale preds so that the class probabs of each sample sum to 1. If final layer outputs have already been passed
        # through softmax, this won't change the values since they sum to 1 already.
        y_pred /= y_pred.sum(axis=-1, keepdims=True)
    # Avoid numerical instability with epsilon clipping. Specifically, categorical cross-entropy uses the
    # log(probability) associated with the true class. If output layer spits out 0-probabilities (as is the case with
    # sparsemax (Martins, André FT, and Ramón Fernandez Astudillo. "From Softmax to Sparsemax: A Sparse Model of
    # Attention and Multi-Label Classification." arXiv preprint arXiv:1602.02068 (2016)), we replace 0 probabilities
    # with some epsilon value.
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)

    cross_ent = -T.mean(T.sum(y_actual * T.log(y_pred),axis=-1), axis=0)
    # inner sum is over class labels
    # outer average is over samples

    # Out of the box theano implementation of categorical cross-entropy. It does not average over samples which makes
    # the learning rate dependent on batch size when optimizing
    # return T.nnet.categorical_crossentropy(y_pred, y_actual)
    return cross_ent


def sparse_categorical_crossentropy_loss(y_pred, y_actual, from_logits=False, **kwargs):
    """
    Expects an array of integer classes.
    Note: labels shape must have the same number of dimensions as y_pred shape.
    If you get a shape error, add a length-1 dimension to labels, for example using np.expand_dims(y, -1)
    """
    print("Using sparse categorical cross-entropy. Ensure y_actual is a vector with the appropriate integer class "
          "labels between 0 and nb_class - 1 (inclusive), but with the same number of dimensions as y_pred. Use "
          "np.expand_dims(y, -1)")
    y_actual = T.cast(T.flatten(y_actual), 'int32')
    y_actual = T.extra_ops.to_one_hot(y_actual, nb_class=y_pred.shape[-1])
    y_actual = T.reshape(y_actual, y_pred.shape)
    return categorical_crossentropy_loss(y_pred, y_actual, from_logits, **kwargs)

def binary_crossentropy_loss(y_pred, y_actual, from_logits=False, **kwargs):
    """
    Assumes inputs are vectors with 0 or 1 values as class labels
    """
    print("Binary cross entropy expects gold labels that are 1 or 0 and model "
          "outputs that are in the range of 0 to 1!")
    assert y_pred.ndim == y_actual.ndim, \
        "Dimensions of inputs to binary cross entropy gold outputs must match!"
    if from_logits:
        output = T.nnet.sigmoid(y_pred)
    else:
        output = y_pred
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, epsilon, 1.0 - epsilon)
    # Averaging across samples
    bin_ce_loss = -T.mean(input=y_actual * T.log(output) + (1 - y_actual) * T.log(1 - output), axis=0)

    # Adding across other dimensions
    if y_pred.ndim > 1:
        bin_ce_loss = T.sum(input=bin_ce_loss)
    return bin_ce_loss
    # return T.nnet.binary_crossentropy(output, y_actual) # Out of the box theano binary cross entropy implementation

def kullback_leibler_divergence_loss(y_pred, y_actual, **kwargs):
    y_actual = T.clip(y_actual, epsilon, 1)
    y_pred = T.clip(y_pred, epsilon, 1)
    return T.sum(y_actual * T.log(y_actual / y_pred), axis=-1)

def poisson_loss(y_pred, y_actual, **kwargs):
    return T.mean(y_pred - y_actual * T.log(y_pred + epsilon), axis=-1)


def cosine_proximity_loss(y_pred, y_actual, **kwargs):
    y_actual = l2_normalize(y_actual, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return -T.mean(y_actual * y_pred, axis=-1)

loss_selector = {fun: globals()[fun] for fun in filter(lambda x: x.endswith("_loss"), globals())}

def get_loss(loss_type, y_pred, y_actual, **kwargs):
    loss_type += "_loss"
    assert loss_type in loss_selector.keys(), "Invalid loss selected. Please select one of " + str(kwargs.keys())
    print("Not all losses are averaged across samples! This could lead to varying sensitivities to learning rates!")
    return(loss_selector[loss_type](y_pred, y_actual, **kwargs))

# IMPLEMENT A MULTICLASS LOSS FUNCTION!!