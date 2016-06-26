import theano
import theano.tensor as T
try:
    from theano.tensor.nnet.nnet import softsign as T_softsign
except ImportError:
    from theano.sandbox.softsign import softsign as T_softsign

def softmax_activation(x, **kwargs):
    # 1-D x is not broadcasted to 2-D to prevent unexpected behaviours or unintended broadcasts
    if x.ndim == 2:
        return(T.nnet.softmax(x))
    elif x.ndim == 3:
        nr = T.exp(x - T.max(x, axis=1, keepdims=True)) # Subtracting the max here prevents huge exponents
        dr = T.sum(nr, axis=-1, keepdims=True) # Since we divide by the sum, the subtraction of max has no effect
        # nr/dr is the expected sotmax of each vector along the last axis
        return(nr/dr)
    else:
        raise Exception("Cannot apply softmax on tensor that is not 2D or 3D" + "Here, ndim = " + str(x.ndim) )

def softplus_activation(x, **kwargs):
    # sigmoid units (binary neurons) are used as binary feature detectors (indicator variables) and model the probability
    # of the feature detector taking on value 1. For problems that need more expressive units (such as modeling real
    # valued outputs), sigmoid units can replicated infinitely with same weight and bias terms, with each replicated
    # unit differing from the previous one by a bias offset of 1. This is the stepped sigmoid function:
    #   sum_{i=1}^{inf} ( sigmoid(x - i + 0.5) ), where sigmoid(x) = 1 / (1 + exp(-x) )
    # Evaluating stepped sigmoid is computationally intensive, so an approximation of it is the softplus function:
    #   log(1 + exp(x_{ij}) evaluted elementwise for each element in x.
    # This converges to 0 asymptotically and obtains a value that is close to zero for even relatively small negative
    # values.
    # Unlike sigmoid, softplus is used to produce any positive real number as the output. It slope is 1 at +inf and 0
    # at -inf. This is in sharp contrast to sigmoid where slope is 0 both at +inf and -inf. Consequently, training tends
    # to be faster with softplus since vanishing gradient for large x (which plagues both sigmoid and tanh) is avoided.
    return(T.nnet.softplus(x))

def relu_activation(x, leak_slope=0., clip_threshold=None, **kwargs):
    # Reference:
    # Nair, Vinod, and Geoffrey E. Hinton. "Rectified linear units improve restricted boltzmann machines."
    # In Proceedings of the 27th International Conference on Machine Learning (ICML-10), pp. 807-814. 2010.
    #
    # softplus in turn can be approximated by a simple max operation max(0, x + N(0, sigmoid(x))). The gaussian noise
    # component is added since softplus behaves like a noisy integer valued version of a smoothed rectified linear unit.
    # The variance of this noise is sigmoid(x) and does not becom large for large x. This can further be simplified by
    # usinf max(0,x) instead. This function is known as Rectified Linear (ReL). This has some advantages:
    #   - No vanishing gradient at +inf, like softplus
    #   - Induces sparsity in activations
    #   - Empirical results indicate deep networks can be trained effectively with ReL units (ReLU)
    #   - Can be used by RBMs to model real/integer valued inputs
    assert hasattr(T.nnet, 'relu'), ('It looks like like your version of '
                                     'Theano is out of date. '
                                     'Install the latest version with:\n'
                                     'pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps')
    assert leak_slope is not None, "Leak slope cannot be None"
    x = T.nnet.relu(x, leak_slope)
    if clip_threshold is not None:
        x = T.minimum(x, clip_threshold)
    return x




def softsign_activation(x, **kwargs):
    # See X. Glorot and Y.
    # Bengio. Understanding the difficulty of training deep feedforward neural
    # networks. In Proceedings of the 13th International Workshop on
    # Artificial Intelligence and Statistics, 2010.
    #   - http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    # This returns the element-wise value of:
    # x / ( 1 + |x| )
    # This has a slope = 1 / ( 1 + |x| )^2
    # Thus softsign also asymptotically approches slope of 0 for both large positive and negative values, but the drop-
    # off is quadratic rather than exponential .
    return(T_softsign(x))

def tanh_activation(x, **kwargs):
    return(T.tanh(x))

def sigmoid_activation(x, **kwargs):
    return(T.nnet.sigmoid(x))

def hard_sigmoid_activation(x, **kwargs):
    return(T.nnet.hard_sigmoid(x))

def linear_activation(x, **kwargs):
    return(x)

activ_selector = {fun: globals()[fun] for fun in filter(lambda x: x.endswith("_activation"), globals())}

def get_activation(activ_type, x, leak_slope=0., clip_threshold=None,**kwargs):
    activ_type += "_activation"
    assert activ_type in activ_selector.keys(), "Invalid activation selected"
    activation_args = {'x':x, 'leak_slope':leak_slope, 'clip_threshold':clip_threshold}
    # assert clip_threshold is None or isinstance(clip_threshold, int),"Invalid clip_threshold specified"
    activ_x = activ_selector[activ_type](**activation_args)
    # if clip_threshold is not None:
    #     activ_x = T.minimum(activ_x, clip_threshold)
    #     print("Currently using T.minimum, but should probably use T.clip instead")
    return(activ_x)