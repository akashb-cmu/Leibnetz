import theano
import numpy as np
import  constraints
from utils import eprint

"""
This package implements various different initialization schemes such as:
1. Fan
2. Uniform
3. Normal
4. Lecun_Uniform
5. Glorot_Uniform
6. Glorot_Normal
7. he_normal_param_init
8. he_uniform_param_init
9. Orthogonal
10. Identity
11. Zeros
12. Ones

For all initializers other than Identity, Zeros and Ones, a numpy.random.RandomState(<seed>) can optionally be passed as
an argument to recreate initializations in case results need to be reproduced.

This module also contains a function that takes a string argument and returns the appropriate initialized value,to avoid
having if-else blocks to do so in every layer that allows choice of initializer.
"""

"""
TO DO :
In get_fans, in dim_ordering == "tf" case, keras implements filter_size = np.prod(dim_tuple[:2])
"""

def get_fans(dim_tuple, dim_ordering="th"):
    if len(dim_tuple) == 2: # Case of input and output dim
        fan_in = dim_tuple[0]
        fan_out = dim_tuple[1]
    elif len(dim_tuple) == 4 or len(dim_tuple) == 5: # 2D and 3D convolution cases repsectively
        # First two/Last two dimensions are (number_of_filters, number_of_input_filter_maps, ...)
        # or (..., number_of_input_filter_maps, number_of_filters)
        # with order specified as 'th' or 'tf' respectively
        # Remaining dimensions specify the (num input feature maps, filter height, filter width (, filter_depth)).
        # Thus the number of parameters is their product.
        # For 2D convolution, the dimensions are (nb_filter, stack_size, nb_row, nb_col)
        # where stack_size is the number of input feature maps, nb_row and nb_col are the filter width and height resp.
        # and NOT the input image width and height.
        if dim_ordering == "th": # Corresponds to theano, where the dims are [nb_filter, stack_size, nb_row, nb_col]
            filter_size = np.prod(dim_tuple[2:])
            fan_in = dim_tuple[1] * filter_size # Each output node processes a region of size=(num_channels*filter_size)
            # i.e. connectivity is local in space (filter_size) but full in depth (num_channels)
            fan_out = dim_tuple[0] * filter_size
            # Dimension of output filter neurons is determined by:
            # (input_image_dim - filter_size_dim + 2*padding_dim)/stride + 1
            # Tot number of output neurons is then dim_1 * dim_2 < * dim_3 > * num_filters
            # Each of these neurons is connected to fan_in sized region in the input. Weights of output neurons at
            # different depths are different but weights of neurons at same depth are shared. Thus, we only need:
            # num_filters * filter_size number of parameters ( + num_filters biases which are initialized separately)
            # The fan_out thus represents the number of unique parameters to be learned.
        elif dim_ordering == "tf": # Corresponds to tensorflow, where the dims are [nb_row, nb_col, stack_size,
            # nb_filter]
            print("Initializer: get_fans in tf mode for 4 or 5 dimensions deviates from Keras implementation. Please recheck implementation")
            filter_size = np.prod(dim_tuple[:-2])
            fan_in = dim_tuple[-2] * filter_size
            fan_out = dim_tuple[-1] * filter_size
        else:
            assert False, "Initializer: Invalid dimension ordering specified"
    else:
        fan_in = np.sqrt(np.prod(dim_tuple))
        fan_out = np.sqrt(np.prod(dim_tuple))
    return fan_in, fan_out

def uniform_param_init(name, dim_tuple, scale=0.05, rnd=None, **kwargs):
    np_mat = np.random.uniform(low=-scale, high=scale, size=dim_tuple) if rnd is None else rnd.uniform(low=-scale, high=scale,
                                                                                              size=dim_tuple)
    return(theano.shared(value=np_mat, name=name, strict=False)) # strict=False => no strong dtype checking is done (no
    # exception in case of mismatch)

def normal_param_init(name, dim_tuple, scale=0.05, rnd=None, **kwargs):
    np_mat = np.random.normal(loc=0.0, scale=scale, size=dim_tuple) if rnd is None else rnd.normal(loc=0.0,
                                                                                                    scale=scale,
                                                                                                    size=dim_tuple)

    return (theano.shared(value=np_mat, name=name, strict=False))

def lecun_uniform_param_init(name, dim_tuple, dim_ordering="th", rnd=None, **kwargs):
    """
    Reference: LeCun 98, Efficient Backprop
        http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    This initialization must be used with training set normalization and choice of a 0 centered sigmoid (includes tanh)
    activation function. LeCun recommends 1.7159 * tanh(2x/3) as an appropriate 0-centered activation. When all three
    are used in conjunction (normalized inputs, 0-centered sigmoid and lecun_uniform_param_init initialization), outputs at all
    nodes of the neural network will have variance of 1 and initial gradients will be in the linear region of the
    sigmoid, which means linear mappings will be learnt first and small gradients at asymptotic extremities of the
    sigmoid will be avoided.
    """
    fan_in, fan_out = get_fans(dim_tuple, dim_ordering)
    scale = np.sqrt(3./fan_in) # LeCun's paper actually recommends sqrt(1/fan_in)
    return(uniform_param_init(name=name, dim_tuple=dim_tuple, scale=scale, rnd=rnd))

def glorot_uniform_param_init(name, dim_tuple, dim_ordering="th", rnd=None, **kwargs):
    fan_in, fan_out = get_fans(dim_tuple, dim_ordering)
    scale = np.sqrt(6./(fan_in + fan_out))
    return(uniform_param_init(name=name, dim_tuple=dim_tuple, scale=scale, rnd=rnd))

def glorot_normal_param_init(name, dim_tuple, dim_ordering="th", rnd=None, **kwargs):
    """
    Reference: Glorot & Bengio, AISTATS 2010
    """
    fan_in, fan_out = get_fans(dim_tuple, dim_ordering)
    scale = np.sqrt(2./(fan_in + fan_out))
    return(normal_param_init(name=name, dim_tuple=dim_tuple, scale=scale, rnd=rnd))

def he_uniform_param_init(name, dim_tuple, dim_ordering="th", rnd=None, **kwargs):
    """
    Reference: Reference:  He et al., http://arxiv.org/abs/1502.01852
    """
    fan_in, fan_out = get_fans(dim_tuple, dim_ordering)
    scale = np.sqrt(6./fan_in)
    return(uniform_param_init(name=name, dim_tuple=dim_tuple, scale=scale, rnd=rnd))

def he_normal_param_init(name, dim_tuple, dim_ordering="th", rnd=None, **kwargs):
    """
    Reference: He et al., http://arxiv.org/abs/1502.01852
    """
    fan_in, fan_out = get_fans(dim_tuple, dim_ordering)
    scale = np.sqrt(2./fan_in)
    return(normal_param_init(name=name, dim_tuple=dim_tuple, scale=scale, rnd=rnd))

def orthogonal_param_init(name, dim_tuple, rnd=None, **kwargs):
    """
    Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    This initialization is good since it can achieve dynamical isometry which can lead to depth independent learning
    times. While this paper does analysis for a deep linear networks, it could potentially be useful even for non-linear
    networks. All layers in the network must be initialized using this method in order to reap any benefits.
    """
    flat_shape = (dim_tuple[0], np.prod(dim_tuple[1:]))
    rnd = np.random if rnd is None else rnd
    a = rnd.normal(loc=0.0, scale=1.0, size=flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one_param_init with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(dim_tuple)
    return(theano.shared(value=q, name=name, strict=False))

def identity_param_init(name, dim_tuple, scale=1.0, **kwargs):
    if len(dim_tuple) != 2 or dim_tuple[0] != dim_tuple[1]:
        raise Exception('Identity matrix initialization can only be used '
                        'for 2D square matrices.')
    else:
        return(theano.shared(value= scale * np.identity(dim_tuple[0], dtype=theano.config.floatX), name=name))

def zero_param_init(name, dim_tuple, **kwargs):
    return(theano.shared(value=np.zeros(dim_tuple, dtype=theano.config.floatX), name=name, strict=False))

def one_param_init(name, dim_tuple, scale=1.0, **kwargs):
    return(theano.shared(value=scale * np.ones(dim_tuple, dtype=theano.config.floatX), name=name, strict=False))

init_selector = {fun: globals()[fun] for fun in filter(lambda x: x.endswith("_param_init"), globals())}

# def get_init_value(init_type, name, dim_tuple, **kwargs):
#     init_type += "_param_init"
#     assert init_type in init_selector.keys(), "Invalid initialization specified: " + init_type
#     return(init_selector[init_type](name, dim_tuple, **kwargs))

def get_init_value(init_type, name, dim_tuple, dim_ordering='th', weights=None, constraint=None, rnd=None, scale=0.1, **kwargs):
    init_type += "_param_init"
    eprint("New initializer with pretrained weights and constraints has not been tested!")
    if constraint is not None:
        assert isinstance(constraint, constraints.Constraint), "Supplied constraint is not an instance of the " \
                                                               "constraint class"
    if weights is not None:
        assert isinstance(weights, np.ndarray), "Pretrained weight supplied must be a numpy array!"
        assert len(dim_tuple) == len(weights.shape) and \
               all(dim == wdim for (dim, wdim) in zip(dim_tuple, weights.shape)), "Dimensions of the supplied weights " \
                                                                                 "don't mach the dim tuple"
        if constraint is not None:
            return theano.shared(value=constraint.np_constrain(weights), name=name, strict=False)
        else:
            theano.shared(value=weights, name=name, strict=False)
    assert init_type in init_selector.keys(), "Invalid initialization specified: " + init_type
    init_args = kwargs
    # init_args = {'name':name, 'dim_tuple':dim_tuple, 'dim_ordering':dim_ordering, 'rnd':rnd, 'scale':scale}
    init_args.update({'name': name, 'dim_tuple': dim_tuple, 'dim_ordering': dim_ordering, 'rnd': rnd, 'scale': scale})
    # return(init_selector[init_type](name=name, dim_tuple=dim_tuple, **kwargs))
    svar = init_selector[init_type](**init_args)
    if constraint is not None:
        svar.set_value(constraint.np_constrain(svar.get_value()))
    return(svar)