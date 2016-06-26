import theano
import theano.tensor as T
import numpy as np
import abc

"""
Regularizer must be passed as an object to a layer
"""

class Regularizer(object):
    __metaclass__ = abc.ABCMeta
    def regularize(self, param, is_train):
        # assert isinstance(param, theano.compile.sharedvalue.SharedVariable), "Providing regularization param is " \
        #                                                                      "not a theano shared variable"
        # Regularizer logic goes here
        pass

class WeightRegularizer(Regularizer):

    def __init__(self, l1_coeff=0.0, l2_coeff=0.0):
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff

    def regularize(self, param, is_train=True):
        # assert isinstance(param, theano.compile.sharedvalue.SharedVariable), "Providing regularization param is " \
        #                                                                      "not a theano shared variable"
        if is_train:
            return(self.l1_coeff * T.sum(T.abs_(param)) + self.l2_coeff * T.sum(T.square(param)))
        else:
            return 0.

class EigenvalueRegularizer(Regularizer):
    """
    Eigenvalue Decay, aims to improve
    the classification margin, which is an effective strategy
    to decrease the classifier complexity, in Vapnik sense, by
    taking advantage on geometric properties of the training examples
    within the feature space.

    Reference: Deep Learning with Eigenvalue Decay Regularizer (https://arxiv.org/pdf/1604.06985v3.pdf)

    Specifically, this paper shows that by minimizing the dominant eigen vector of W * W' we CAN increase the lower
    bound on the classification margin for any arbitrary training sample towards its correct class. Note that this is
    not guaranteed since the margin is a function of the weight matrix itself, but the terms involving the dominant
    eigen vector all appear in the denominator.
    """
    def __init__(self, reg_coeff=0., power_iter=9):
        self.power_iter = power_iter
        self.reg_coeff = reg_coeff

    def regularize(self, param, is_train=True):
        if is_train:
            assert param.ndim == 2, "Eigen value regularization is currently supported only for 2D parameter matrices"
            # assert isinstance(param, theano.compile.sharedvalue.SharedVariable), "Providing regularization param is " \
            #                                                                      "not a theano shared variable"
            print("Using Eigen Value Decomposition as a regularization technique. This can significantly slow down "
                  "optimization due to power iteration technique for evaluating dominant eigen value in a differentiable"
                  "fashion.")
            try:
                param.get_value()
            except Exception as e:
                print("param is not a theano shared variable. Cannot regularize using Eigen value decay since "
                      "dimensions of weight matrix must be known.")
                print(str(e))
            M = T.dot(T.transpose(param), param)
            eig_0 = theano.shared(np.ones(shape=(M.shape.eval()[0],1), dtype=theano.config.floatX))
            # try:
            #     import theano.tensor.slinalg.Expm as TExpm
            #     M_p = TExpm(M, self.power_iter)
            # except:
            #     print("Your version of theano doesn't support matrix exponentiation. Falling back to iterative "
            #           "implementation")
            domin_eig = T.dot(M, eig_0)
            for i in range(self.power_iter - 1):
                domin_eig = T.dot(M, domin_eig)

            M_d = T.dot(M, domin_eig)
            dom_eig_val = T.dot(T.transpose(M_d), domin_eig) / T.dot(T.transpose(domin_eig), domin_eig)
            regularized_loss = T.sqrt(dom_eig_val) * self.reg_coeff
            # The regularized loss is an array with a single value. So use T.sum to collapse into a scalar
            regularized_loss = T.sum(regularized_loss)
            return( regularized_loss )
        else:
            return(0.)

"""
Tips for choosing l1 regularization constant: ---> Applies for linear regression
Assuming a whitening transform is applied on the inputs i.e. X'.X = I, the weight for the i^th feature in l1
regularization is 0 if |y' . X_{.i}| < reg_coeff, where X_{.i} is the vector of feature values for the i^th feature
across all samples and y is the vector of gold prediction values corresponding t each of those samples.
"""


def l1(l1_reg_coeff=0.01):
    return(WeightRegularizer(l1_coeff=l1_reg_coeff, l2_coeff=0.0))

def l2(l2_reg_coeff=0.01):
    return(WeightRegularizer(l1_coeff=0.0, l2_coeff=l2_reg_coeff))

def l1l2(l1_reg_coeff=0.01, l2_reg_coeff=0.01):
    return(WeightRegularizer(l1_coeff=l1_reg_coeff, l2_coeff=l2_reg_coeff))

def eigen_reg(eig_reg_coeff=0.01, power_iters=9):
    return(EigenvalueRegularizer(reg_coeff=eig_reg_coeff, power_iter=power_iters))