import theano
from theano import tensor as T
from theano import config as Tconfig
import abc
import initializers
import activations
import regularizers
import constraints
import numpy as np
from model import Component


# In order for any type of layer to be used in a model, it MUST subclass from Component, since doing so ensures the
# layer implements all functionalities required to build, compile, train and test a model.

class Layer(Component):
    """
    THIS DOCUMENTATION IS OUTDATED! PLEASE UPDATE!

    This is an abstract class that must be inherited to implement a new layer. It encapsulates the following data and
    methods.

    TO DO:
    Support masking

    1. Data:
        - name
        - input_dim
        - inputs # List of input layers
        - rnd_seed ---> may not be required
        - input_dtype
        - output_dim
        - output_dtype
        - constraints
        - regularizers
        - activation
        - init_type
        - trainable_params
        - trainable_param_names
        - fixed_params
        - fixed_param_names
        - l_rate
        - leak_slope # Relevant for relu only
        - clip_threshold ---> may not be required
        - dropout_p # for dropout layers only
        - trainable
        - built

    2. Methods:
        - link_to_inputs() : Accepts input tensor and applies the layer logic
        - get_train_output() : Returns the output tensor to be used at train time
        - get_test_output() : Returns the output tensor to be used at test time
        - get_trainable_params() : Returns (param, param_name) for all trainable parameters
        - get_fixed_params() : Returns (param, param_name) for all non-trainable parameters
        - build_all() :
        - set_params() : Set weights of the layer using a provided dictionary
        - get_output_shape() : Get the shape of the layer's output
        - get_config() : Get this layers configurations
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        """
        Initializes various class members to defaults which are to be specified by child classes. Sanity checks also
        performed.
        """
        super(Layer, self).__init__()
        required_args = {"name", "output_dim", "input_dim"}
        required_args_str = " ".join([arg_str for arg_str in required_args])
        assert all(arg in kwargs.keys() for arg in required_args), "All required args not specified. Please " \
                                                                          "specify " + required_args_str
        self.layer_name = kwargs["name"]
        self.component_name = self.layer_name
        self.input_dim = kwargs["input_dim"]
        self.output_dim = kwargs["output_dim"]
        self.input_dtype = kwargs.get("input_dtype", Tconfig.floatX)
        # self.output_dtype = Tconfig.floatX
        self.output_dtype = kwargs.get("output_dtype",Tconfig.floatX)
        # self.trainable_params = {} # Must be specified by child class implementation
        # self.trainable_param_names = [] # Must be specified by child class implementation
        self.fixed_params = {} # Must be specified by child class implementation
        self.fixed_param_names = [] # Must be specified by child class implementation
        # self.regularizers = {} # Must be specified by child class implementation based on kwargs
        # self.constraints = {} # Dict that maps tensor: constraints instance. Must be specified by child class
        self.activation = kwargs.get("activation", None)
        self.init_type = kwargs.get("init_type", "glorot_uniform")
        self.learning_rate = kwargs.get("learning_rate", 0.01)
        self.leak_slope = kwargs.get("leak_slope", 0.01)
        self.rnd_seed = kwargs.get("rnd_seed", None)
        self.clip_threshold = kwargs.get("clip_threshold", None)
        self.trainable = kwargs.get("trainable", True)
        self.rnd = None
        # self.trainable = kwargs.get("trainable", True) # May be relevant for something like a fixed embedding layer
        # which need not be trainable but still has some parameters (albeit fixed parameters)

    # link_to_inputs assumes a single path from input to output.

    # @abc.abstractmethod
    # def link_to_inputs(self, is_train=True):
    #     # Recursively calls itself on input layers to current layer to obtain the output tensor in a top-down fashion.
    #     # In addition to the current layer, this function assumes all input layers configured for the current layer have
    #     # their input layers fixed as well.
    #     pass

    def link(self, input, is_train=True):
        # Takes a provided input tensor and applies current layer's logic on top of it. Differs from link_to_inputs in
        # that it doesn't place recursive calls and doesn't expect all input layers (or even the current layer) to have
        # already fixed inputs.
        assert isinstance(input, theano.tensor.TensorVariable), "Input tensor must be a theano tensor " \
                                                                        "variable (theano.tensor.TensorVariable)"
        if is_train:
            return (self.get_train_output(input))
        else:
            return (self.get_test_output(input))

    @abc.abstractmethod
    def get_train_output(self, input):  # relevant for things like dropout or noise layers which modify input during
        # train time
        pass

    @abc.abstractmethod
    def get_test_output(self, input):
        # relevant for things like dropout or noise layers which need to adjust input during
        # test time
        pass


    def get_fixed_params(self):
        # Return the appropriate fixed weights along with their names as zipped parallel lists
        return(self.fixed_params)

    @abc.abstractmethod
    def set_params(self, weights_dict):
        # Using the dictionary of weights provided to set parameters
        pass

    def get_output_shape(self, input_shape):
        # Return the appropriate output shape
        assert input_shape and len(input_shape) == 2, "Provide input with shape (n_input_samples, input_dim)"
        return ((input_shape[0], self.output_dim))

    @abc.abstractmethod
    def get_config(self):
        # Return this layer's configuration
        return

    # The following get_all_regularization_costs assumes a unique path from input to output

    # @abc.abstractmethod
    # def get_all_regularization_costs(self, is_train=True):
    #     # Get the regularization costs for this layer and all input-layers recursively
    #     pass

    # @abc.abstractmethod
    # def get_gradients(self, cost):
    #     # This method takes a cost theano variable and evaluates the updates for all the parameters in this layer
    #     # with respect to this cost. Note that this method does NOT take care of adding the regularization cost of
    #     # parameters in this layer. Rather it assumes addition of regularization costs have already been added to the
    #     # input cost variable.
    #     pass

class DenseLayer(Layer):

    """
    In addition to the default parameters, this layer uses the following parameters:
    - use_biases
    - rnd_seed
    - W_regularizer
    - W_constraint
    - b_regularizer
    - b_constraint
    - weights
    - biases
    - clip_threshold
    """

    def __init__(self, name, output_dim, input_dim, input_dtype=Tconfig.floatX, output_dtype=Tconfig.floatX,
                 activation='linear', init_type='glorot_uniform',
                 learning_rate=0.01, leak_slope=0.01, clip_threshold=None, weights=None, biases=None, W_regularizer=None,
                 W_constraint=None, b_regularizer=None, b_constraint=None, use_bias=True, rnd_seed=None, trainable=True,
                 **kwargs):
        super_args = {key: value for (key, value) in kwargs.items()}
        local_args = {key: value for (key, value) in locals().items()}
        assert not (not use_bias and biases is not None), "Not using bias but bias weights have been provided."
        del local_args['self']
        del local_args['kwargs']
        super_args.update(local_args)
        del local_args
        super(DenseLayer, self).__init__(**super_args)
        self.clip_threshold = clip_threshold
        self.use_bias = use_bias
        self.w_name = self.layer_name + "_W"
        self.b_name = self.layer_name + "_b"
        self.W = None
        self.b = None
        w_dim_tuple = (self.input_dim, self.output_dim)
        b_dim_tuple = (self.output_dim,)
        if weights is None:
            if self.rnd_seed is None:
                self.rnd = np.random.RandomState()
            else:
                self.rnd = np.random.RandomState(self.rnd_seed)
        else:
            self.rnd_seed = None

        if W_regularizer is not None:
            assert isinstance(W_regularizer, regularizers.Regularizer), "Specified weight regularizer is not instance " \
                                                                        "" \
                                                                        "of Regularizer class"
            self.regularizers[self.w_name] = W_regularizer
        if b_regularizer is not None:
            assert self.use_bias, "Can't set regularizer for bias since use_bias is False"
            assert isinstance(b_regularizer, regularizers.Regularizer), "Specified bias regularizer is not instance " \
                                                                        "" \
                                                                        "of Regularizer class"
            self.regularizers[self.b_name] = b_regularizer
        if W_constraint is not None:
            assert isinstance(W_constraint, constraints.Constraint), "Specified weight constraint is not instance of" \
                                                                     "Constraint class"
            self.constraints[self.w_name] = W_constraint
        if b_constraint is not None:
            assert self.use_bias, "Can't set constraint for bias since use_bias is False"
            assert isinstance(b_constraint, constraints.Constraint), "Specified bias constraint is not instance of" \
                                                                     "Constraint class"
            self.constraints[self.b_name] = b_constraint

        # self.activation = activation # Already set by parent class
        # self.trainable = True # Default True is set in Layer

        # ALWAYS ENSURE ALL CONFUGRATION TYPE INITIALIZATIONS ARE DONE BEFORE WEIGHT INITIALIZATIONS ARE DONE!!
        # FOR EXAMPLE, EARLIER  THE CONTRAINT INITIALIZATION WAS DONE AFTER WEIGHT AND BIAS INITIALIZATION SO
        # CONSTRAINTS WERE NOT APPLIED TO THE INITIALIZED VALUES

        if weights is not None:
            weights = np.array(weights)
            assert len(weights.shape) == len(w_dim_tuple) and all(weights.shape[i] == w_dim_tuple[i] for
                   i in range(len(weights.shape))), "Provided weights of shape " + str(weights.shape) + \
                                                    " don't match the input and output shape " + w_dim_tuple
            self.set_params(W=weights)
        else:
            self.set_params(W=initializers.get_init_value(init_type=self.init_type, name=self.w_name,
                                                          dim_tuple=w_dim_tuple, rnd=self.rnd))
        assert not(biases is not None and not self.use_bias), "Bias weights specified even when not using bias!"
        if self.use_bias:
            if biases is not None:
                biases = np.array(biases)
                assert len(biases.shape) == len(b_dim_tuple) and all(biases.shape[i] == b_dim_tuple[i] for
                       i in range(len(biases.shape))), "Provided biases of shape " + str(biases.shape) + \
                                                        " don't match the output shape " + b_dim_tuple
                self.set_params(b=biases)
            else:
                self.set_params(b=initializers.get_init_value(init_type="zero", name=self.b_name, dim_tuple=b_dim_tuple))

        

    def get_train_output(self, input):
        scores = T.dot(input, self.W)
        if self.use_bias:
            scores += self.b
        output = activations.get_activation(activ_type=self.activation, x=scores,leak_slope=self.leak_slope,
                                            clip_threshold=self.clip_threshold)
        return(output)

    def get_test_output(self, input):
        return(self.get_train_output(input=input))

    def set_params(self, W=None, b=None): # accepts np.ndarray matrix/vector or theano shared variable
        if W is not None:
            if isinstance(W, theano.tensor.sharedvar.TensorSharedVariable):
                weights = self.constraints[self.w_name].np_constrain(W.get_value()) if self.constraints.get(self.w_name,
                          None) is not None else W.get_value()
                assert len(weights.shape) == 2 and weights.shape[0] == self.input_dim and weights.shape[1] == \
                self.output_dim, "Provided weights with shape = " + str(weights.shape) + " for " + self.layer_name + \
                " are not compatible with the layer shape " + str((self.input_dim, self.output_dim))
                if self.W is not None:
                    self.W.set_value(new_value=weights)
                else:
                    self.W = theano.shared(value=weights,name=self.w_name,strict=False)
            elif isinstance(W, np.ndarray):
                W = self.constraints[self.w_name].np_constrain(W) if self.constraints.get(self.w_name, None) is not None\
                                                                  else W
                assert len(W.shape) == 2 and W.shape[0] == self.input_dim and W.shape[1] == self.output_dim, \
                    "Provided weights with shape = " + str(W.shape) + " for " + self.layer_name + " are not compatible"\
                    " with the layer shape " + str((self.input_dim, self.output_dim))
                if self.W is not None:
                    self.W.set_value(new_value=W)
                else:
                    self.W = theano.shared(value=W,name=self.w_name,strict=False)
            else:
                assert False, "set_params() expects weights to be either a theano shared variable or numpy.ndarray" \
                              "matrix"
            self.trainable_params[self.w_name] = self.W
        if b is not None:
            assert self.use_bias, "Layer configured to not use bias. Cannot set bias unless this is changed."
            if isinstance(b, theano.tensor.sharedvar.TensorSharedVariable):
                biases = self.constraints[self.b_name].np_constrain(b.get_value()) if self.constraints.get(self.b_name,
                         None) is not None else b.get_value()
                assert len(biases.shape) == 1 and biases.shape[0] == \
                    self.output_dim, "Provided biases with shape = " + str(biases.shape) + " for " + self.layer_name + \
                    " are not compatible with the layer shape " + str((self.input_dim, self.output_dim))
                if self.b is not None:
                    self.b.set_value(new_value=biases)
                else:
                    self.b = theano.shared(value=biases, name=self.b_name, strict=False)
            elif isinstance(b, np.ndarray):
                biases = self.constraints[self.b_name].np_constrain(b) if self.constraints.get(self.b_name, None) is \
                                                                                      not None else b
                assert len(biases.shape) == 1 and biases.shape[0] == \
                                                  self.output_dim, "Provided biases with shape = " + str(
                    biases.shape) + " for " + self.layer_name + \
                                                                   " are not compatible with the layer shape " + str(
                    (self.input_dim, self.output_dim))
                if self.b is not None:
                    self.b.set_value(new_value=biases)
                else:
                    self.b = theano.shared(value=biases,name=self.b_name,strict=False)
            else:
                assert False, "set_weights() expects bias to be either a theano shared variable or numpy.ndarray" \
                              "vector"
            self.trainable_params[self.b_name] = self.b
        self.trainable_param_names = self.trainable_params.keys()

    def get_config(self):
        config_dict = {
            'name'                 : self.layer_name,
            'input_dim'            : self.input_dim,
            # 'inputs'               : self.inputs,
            'rnd_seed'             : self.rnd_seed,
            'input_dtype'          : self.input_dtype,
            'output_dim'           : self.output_dim,
            'constraints'          : self.constraints,
            'regularizers'         : self.regularizers,
            'activation'           : self.activation,
            'init_type'            : self.init_type,
            'trainable_params'     : self.trainable_params,
            'trainable_param_names': self.trainable_param_names,
            'l_rate'               : self.learning_rate,
            'leak_slope'           : self.leak_slope,
            'clip_threshold'       : self.clip_threshold,
            'trainable'            : self.trainable,
        }
        return(config_dict)

class EmbeddingLayer(Layer):
    """
        By default, int32 is assumed for 1-hot embeddings.
    """
    def __init__(self, layer_name, vocab_size, embed_size, output_dtype=T.config.floatX,
                 init_type='glorot_uniform', learning_rate=0.01, clip_threshold=None, pretrained_embeddings=None,
                 embedding_constraint=None, embedding_regularizer=None, rnd_seed=None, trainable=True, **kwargs):
        input_dtype = 'int32',
        input_dim=vocab_size
        output_dim = embed_size
        name = layer_name
        super_args = {key: value for (key, value) in kwargs.items()}
        local_args = {key: value for (key, value) in locals().items()}
        del local_args['self']
        del local_args['kwargs']
        super_args.update(local_args)
        del local_args
        super(EmbeddingLayer, self).__init__(**super_args)
        self.embeddings_name = self.layer_name + "_embeddings"
        self.embeddings = None
        self.vocab_size = vocab_size
        self.embedding_size = embed_size
        embeddings_dim_tuple = (self.vocab_size, self.embedding_size)
        self.clip_threshold = clip_threshold

        # set constraints and regularizers first
        if pretrained_embeddings is None:
            if self.rnd_seed is None:
                self.rnd = np.random.RandomState()
            else:
                self.rnd = np.random.RandomState(self.rnd_seed)
        else:
            self.rnd_seed = None # Since random seed was not used at all

        if embedding_regularizer is not None:
            assert isinstance(embedding_regularizer, regularizers.Regularizer), "Provided regularizer for embeddings" \
                                                                                " must be instance of Regularizer class"
            self.regularizers[self.embeddings_name] = embedding_regularizer
        if embedding_constraint is not None:
            assert isinstance(embedding_constraint, constraints.Constraint), "Provided constraint must be an instance " \
                                                                             "of Constraint class"
            self.constraints[self.embeddings_name] = embedding_constraint

        self.trainable = trainable
        if pretrained_embeddings is not None:
            pretrained_embeddings = np.array(pretrained_embeddings)
            self.set_params(embed_mat=pretrained_embeddings)
        else:
            self.set_params(embed_mat=initializers.get_init_value(init_type=self.init_type, name=self.embeddings_name,
                                                                  dim_tuple=embeddings_dim_tuple, rnd=self.rnd))

    def link(self, input, is_train=True):
        assert isinstance(input, T.TensorVariable),"Provided input to embedding layer must be a theano tensor"
        assert input.dtype in T.int_dtypes, "Input must be of an integer type. Each integer will be expanded into an" \
                                            " embedding vector"
        return self.get_train_output(input=input) if is_train else self.get_test_output(input=input)

    def get_train_output(self, input): # Assumes a tensor of integers
        return self.embeddings[input]

    def get_test_output(self, input):
        return self.get_train_output(input=input)

    def set_params(self, embed_mat):
        if isinstance(embed_mat, theano.tensor.sharedvar.TensorSharedVariable):
            embeddings = self.constraints[self.embeddings_name].np_constrain(embed_mat.get_value())\
                         if self.constraints.has_key(self.embeddings_name) else embed_mat.get_value()
            assert len(embeddings.shape) == 2 and embeddings.shape[0] == self.vocab_size and\
                   embeddings.shape[1] == self.embedding_size, "Provided embeddings of shape " + str(embeddings.shape)\
                                          + " don't match layer config: " + str((self.vocab_size, self.embedding_size))
            if self.embeddings is None:
                self.embeddings = theano.shared(name=self.embeddings_name, value=embeddings, strict=False)
            else:
                self.embeddings.set_value(new_value=embeddings)
        elif isinstance(embed_mat, np.ndarray):
            embeddings = self.constraints[self.embeddings_name].np_constrain(embed_mat) if \
                         self.constraints.has_key(self.embeddings_name) else embed_mat
            assert len(embeddings.shape) == 2 and embeddings.shape[0] == self.vocab_size and \
                   embeddings.shape[1] == self.embedding_size, "Provided embeddings of shape " + str(embeddings.shape) \
                                                               + " don't match layer config: " + str(
                (self.vocab_size, self.embedding_size))
            if self.embeddings is None:
                self.embeddings = theano.shared(name=self.embeddings_name, value=embeddings, strict=False)
            else:
                self.embeddings = self.embeddings.set_value(new_value=embeddings)
        else:
            assert False, "set_params() expects the pretrained embeddings to be a theano shared variable or a " \
                          "numpy array"
        if self.trainable:
            self.trainable_params[self.embeddings_name] = self.embeddings
        self.trainable_param_names = self.trainable_params.keys()

    def get_config(self):
        config_dict = {
            'name'                 : self.layer_name,
            'input_dim'            : self.input_dim,
            'vocab_size'           : self.vocab_size,
            'embedding_size'       : self.embedding_size,
            'rnd_seed'             : self.rnd_seed,
            'input_dtype'          : self.input_dtype,
            'output_dim'           : self.output_dim,
            'constraints'          : self.constraints,
            'regularizers'         : self.regularizers,
            'init_type'            : self.init_type,
            'trainable_params'     : self.trainable_params,
            'trainable_param_names': self.trainable_param_names,
            'l_rate'               : self.learning_rate,
            'leak_slope'           : self.leak_slope,
            'clip_threshold'       : self.clip_threshold,
            'trainable'            : self.trainable,
        }
        return config_dict