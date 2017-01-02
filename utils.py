from __future__ import print_function
import theano
from theano import tensor as T
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

epsilon = 10e-8

def l2_normalize(x, axis):
    norm = T.sqrt(T.sum(T.square(x), axis=axis, keepdims=True))
    return x / (norm + epsilon) # To avoid division by 0. Even in case x is all zeros, result won't change


# OVERWRITE MEAN AND OTHER SUCH OPERATIONS TO TAKE CARE OF DTYPE CASTING. SEE THE LINEAR ALGEBRA SECTION IN THIS SCRIPT:
# https://github.com/fchollet/keras/blob/master/keras/backend/theano_backend.py
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
