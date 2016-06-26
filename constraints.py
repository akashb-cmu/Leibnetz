import theano
import theano.tensor as T
import numpy as np
import abc

from utils import *

class Constraint(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def constrain(self, param):
        # Return theano tensor parameter after enforcing the constraints
        pass

    @abc.abstractmethod
    def np_constrain(self, np_param):
        # Return numpy parameter after enforcing the constraints
        pass

class MaxNorm(Constraint):
    """
    Constrain the weights incident to each hidden unit to have a norm less than or equal to a desired value. This is
    done by projecting the weight vector onto a ball of the desried max value radius, whenever the norm exceeds this max
    value. Mathematically, this simple means multiplying the parameters by min(max_norm, actual_norm) / (actual_norm)
        # Arguments
            m: the maximum norm for the incoming weights.
            axis: integer, axis along which to calculate weight norms. For instance,
                in a `Dense` layer the weight matrix has shape (input_dim, output_dim),
                set `axis` to `0` to constrain each weight vector of length (input_dim).
                In a `MaxoutDense` layer the weight tensor has shape (nb_feature, input_dim, output_dim),
                set `axis` to `1` to constrain each weight vector of length (input_dim),
                i.e. constrain the filters incident to the `max` operation.
                In a `Convolution2D` layer with the Theano backend, the weight tensor
                has shape (nb_filter, stack_size, nb_row, nb_col), set `axis` to `[1,2,3]`
                to constrain the weights of each filter tensor of size (stack_size, nb_row, nb_col).
                In a `Convolution2D` layer with the TensorFlow backend, the weight tensor
                has shape (nb_row, nb_col, stack_size, nb_filter), set `axis` to `[0,1,2]`
                to constrain the weights of each filter tensor of size (nb_row, nb_col, stack_size).
        # References
            - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014]
              (http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
    """

    def __init__(self, max_norm=2., axis=0):
        self.max_norm = max_norm
        self.axis = axis # Can be a scalar or a list

    def constrain(self, param):
        norm = T.sqrt(T.sum(T.square(param),axis=self.axis, keepdims=True))
        desired_norm = T.clip(norm,0., self.max_norm)
        projection_scaling = desired_norm/(epsilon + norm)
        param *= projection_scaling
        return(param)

    def np_constrain(self, np_param):
        # Applies constraint on the actual numpy array rather than theano tensor
        norm = np.sqrt(np.sum(np.square(np_param), axis=self.axis, keepdims=True))
        desired_norm = np.clip(norm,0.,self.max_norm)
        projection_scaling = desired_norm/(epsilon + norm)
        np_param *= projection_scaling
        return(np_param)

class NonNeg(Constraint):
    """
    Simply zeroes out any negative weights and retains positive ones as is
    """

    def constrain(self, param):
        param *= T.cast(param >= 0., dtype=theano.config.floatX)
        return(param)

    def np_constrain(self, np_param):
        # Applies constraint on the actual numpy array rather than theano tensor
        np_param *= np_param >= 0.
        return(np_param)

class UnitNorm(Constraint):
    """
    # Arguments
        axis: integer, axis along which to calculate weight norms. For instance,
            in a `Dense` layer the weight matrix has shape (input_dim, output_dim),
            set `axis` to `0` to constrain each weight vector of length (input_dim).
            In a `MaxoutDense` layer the weight tensor has shape (nb_feature, input_dim, output_dim),
            set `axis` to `1` to constrain each weight vector of length (input_dim),
            i.e. constrain the filters incident to the `max` operation.
            In a `Convolution2D` layer with the Theano backend, the weight tensor
            has shape (nb_filter, stack_size, nb_row, nb_col), set `axis` to `[1,2,3]`
            to constrain the weights of each filter tensor of size (stack_size, nb_row, nb_col).
            In a `Convolution2D` layer with the TensorFlow backend, the weight tensor
            has shape (nb_row, nb_col, stack_size, nb_filter), set `axis` to `[0,1,2]`
            to constrain the weights of each filter tensor of size (nb_row, nb_col, stack_size).
    """

    def __init__(self, axis=0):
        self.axis = axis

    def constrain(self, param):
        return(param / (epsilon + T.sqrt(T.sum(T.square(param), axis=self.axis, keepdims=True))))

    def np_constrain(self, np_param):
        return(np_param / (epsilon + np.sqrt(np.sum(np.square(np_param), axis=self.axis, keepdims=True))))

def max_norm_constraint(max_norm=2., axis=0, **kwargs):
    return(MaxNorm(max_norm=max_norm,axis=axis))

def non_neg_constraint(**kwargs):
    return(NonNeg())

def unit_norm_constraint(axis=0, **kwargs):
    return(UnitNorm(axis=axis))

constraint_selector = {fun: globals()[fun] for fun in filter(lambda x: x.endswith("_constraint"), globals())}

def get_constraint(constraint_type, **kwargs):
    constraint_type += "_constraint"
    return(constraint_selector[constraint_type](**kwargs))