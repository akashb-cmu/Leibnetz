import theano
from theano import tensor as T
from theano import config as Tconfig
import abc
from utils import eprint
import initializers
import activations
import regularizers
import constraints
import numpy as np
from model import Component

# As was the case with layers, recurrent layers must subclass from component to be used as part of a model

class RecurrentLayer(Component):

    __metaclass__ = abc.ABCMeta # RecurrentLayer is an abstract class that can't be instantiated

    def __init__(self, **kwargs):
        super(RecurrentLayer, self).__init__() # Calling init from the component class
        required_args = {"name", "hidden_dim", "input_dim"}
        assert all(kwargs.has_key(arg) for arg in required_args), "All of the following arguments must be specified: "\
                                                                + " ".join([arg for arg in required_args])
        self.layer_name = kwargs["name"]
        self.component_name = self.layer_name
        self.input_dim = kwargs["input_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.input_dtype = kwargs.get("input_dtype", Tconfig.floatX)
        self.hidden_dtype = kwargs.get("hidden_dtype", Tconfig.floatX)
        self.with_batch = kwargs.get("with_batch", False)

        # self.trainable_params = {} # Initialized by Component but must be specified by child class implementation
        # self.trainable_param_names = [] # Initialized by Component but must be specified by child class implementation
        self.fixed_params = {}  # Must be specified by child class implementation
        self.fixed_param_names = []  # Must be specified by child class implementation
        # self.regularizers = {} # Initialized by Component but must be specified by child class implementation based
                                 # on kwargs
        # self.constraints = {} # Initialized by Component but must be specified by child class/ It is a dict that maps
                                # tensor: constraints instance. Must be specified by child class
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

    def link(self, input, init_hidden=None, is_train=True):
        assert isinstance(input, theano.tensor.TensorVariable), "Input tensor must be a theano tensor " \
                                                                "variable (theano.tensor.TensorVariable)"
        assert init_hidden is None or  isinstance(init_hidden, theano.tensor.TensorVariable), "Init hidden must be a theano tensor " \
                                                                "variable (theano.tensor.TensorVariable)"
        if is_train:
            return (self.get_train_output(input, init_hidden))
        else:
            return (self.get_test_output(input, init_hidden))

    @abc.abstractmethod
    def get_train_output(self, input, init_hidden): # Relevant if you want to build in a dropout functionality for example, in which
        # case you get different outputs at train and test
        pass

    @abc.abstractmethod
    def get_test_output(self, input, init_hidden):
        pass

    @abc.abstractmethod
    def set_params(self, weights_dict):
        pass

    @abc.abstractmethod
    def get_config(self):
        pass

    def get_fixed_params(self):
        return self.fixed_params

    def get_hidden_shape(self, input_shape):
        if (self.with_batch):
            assert (input_shape and len(input_shape) == 3),"With batched recurrent layers, the input must have 3 " \
                   "dimensions corresponding to (batch_size, sequence_len, input_dim)!"
            return (input_shape[0],input_shape[1], self.hidden_dim)
        else:
            assert (input_shape and len(input_shape) == 2),"With NON-batched recurrent layers, the input must have 2 " \
                   "dimensions corresponding to (sequence_len, input_dim)!"
            return (input_shape[0], self.hidden_dim)

class RNN(RecurrentLayer):
    def __init__(self, name, hidden_dim, input_dim, input_dtype=Tconfig.floatX, hidden_dtype=Tconfig.floatX,
                 activation='relu', init_type='glorot_uniform', with_batch=False,
                 learning_rate=0.01, leak_slope=0.01, clip_threshold=5.0, weights_ih=None, weights_hh=None,
                 biases_h=None, W_ih_regularizer=None, W_hh_regularizer=None, W_ih_constraint=None,
                 W_hh_constraint=None, b_h_regularizer=None, b_h_constraint=None,
                 use_bias_h=True, rnd_seed=None, trainable=True, **kwargs):
        super_args = {key:value for (key, value) in kwargs.items()} # To avoid passing kwargs as a dictionary to super
        # constructor
        local_args = {key:value for (key, value) in locals().items()} # This should catch all arguments input to this
        # constructor, but it includes self
        del local_args['self']
        del local_args['kwargs']
        super_args.update(local_args)
        del local_args
        super(RNN, self).__init__(**super_args)
        self.clip_threshold = clip_threshold
        self.use_bias_h = use_bias_h
        self.w_ih_name = self.layer_name + "_W_ih"
        self.w_hh_name = self.layer_name + "_W_hh"
        self.b_h_name = self.layer_name + "_b_h"
        # the initial hidden state is zero by default. However, a different initial state can be provided to the link
        # function
        self.W_ih = None
        self.W_hh = None
        self.b_h = None
        self.w_ih_dim_tuple = (self.input_dim, self.hidden_dim)
        self.w_hh_dim_tuple = (self.hidden_dim, self.hidden_dim)
        self.b_h_dim_tuple = (self.hidden_dim,)
        if weights_hh is None or weights_ih is None:
            if self.rnd_seed is None:
                self.rnd = np.random.RandomState()
            else:
                self.rnd = np.random.RandomState(self.rnd_seed)
        else:
            self.rnd = None

        for (constr, name) in zip([W_ih_constraint, W_hh_constraint, b_h_constraint if use_bias_h else None],
                                  [self.w_ih_name, self.w_hh_name, self.b_h_name]):
            if(constr is not None):
                assert isinstance(constr, constraints.Constraint), "Specified " + name + " constraint must be an " \
                                                                            "instance of the Constraint class!"
                self.constraints[name] = constr

        for (reg, name) in zip([W_ih_regularizer, W_hh_regularizer, b_h_regularizer if use_bias_h else None],
                                  [self.w_ih_name, self.w_hh_name, self.b_h_name]):
            if (reg is not None):
                assert isinstance(reg, regularizers.Regularizer), "Specified " + name + " regularizer must be an " \
                                                                    "instance of the Regularizer class!"
                self.regularizers[name] = reg

        weights_hh = np.ndarray(weights_hh) if (weights_hh is not None and isinstance(weights_hh, list)) else weights_hh

        weights_ih = np.ndarray(weights_ih) if (weights_ih is not None and isinstance(weights_ih, list)) else weights_ih

        assert not( (not use_bias_h) and biases_h is not None), "Pretrained bias weights provided to layer that " \
                                                    "doesn't use biases!"

        biases_h = np.ndarray(biases_h) if (biases_h is not None and isinstance(biases_h, list)) else biases_h

        self.set_params(W_ih=weights_ih, W_hh=weights_hh, b_h=biases_h, init_w=True, init_b=use_bias_h)

        default_init_hid_dim_tup = (hidden_dim,)
        self.default_init_hid = initializers.get_init_value(init_type='zero', name=self.layer_name + "_default_hid_init",
                                                        dim_tuple=default_init_hid_dim_tup)


    def set_params(self, W_ih=None, W_hh=None, b_h=None, init_w=False, init_b=False):

        weights = None
        if W_ih is None:
            if init_w:
                weights = initializers.get_init_value(init_type=self.init_type, name=self.w_ih_name,
                                                  dim_tuple=self.w_ih_dim_tuple, rnd=self.rnd).get_value()
                weights = self.constraints[self.w_ih_name].np_constrain(weights)\
                if self.constraints.get(self.w_ih_name,None) is not None else weights
        elif(isinstance(W_ih, theano.tensor.sharedvar.TensorSharedVariable)):
            weights = self.constraints[self.w_ih_name].np_constrain(W_ih.get_value()) \
                        if self.constraints.get(self.w_ih_name, None) is not None else W_ih.get_value()
        elif (isinstance(W_ih, np.ndarray)):
            weights = self.constraints[self.w_ih_name].np_constrain(W_ih) \
                if self.constraints.get(self.w_ih_name, None) is not None else W_ih
        else:
            assert False, "Provided pretrained value for W_ih must be either a tensor shared variable or a " \
                          "numpy ndarray"
        if weights is not None:

            assert len(weights.shape) == 2 and weights.shape[0] == self.input_dim and \
                   weights.shape[1] == self.hidden_dim, "Provided pretrained W_ih weights are not of the " \
                                                        "appropriate shape!"

            if self.W_ih is not None:
                self.W_ih.set_value(weights)
            else:
                self.W_ih = theano.shared(value=weights, name=self.w_ih_name, strict=False)
        if self.W_ih is not None:
            self.trainable_params[self.w_ih_name] = self.W_ih

        weights = None

        if W_hh is None:
            if init_w:
                weights = initializers.get_init_value(init_type=self.init_type, name=self.w_hh_name,
                                                  dim_tuple=self.w_hh_dim_tuple, rnd=self.rnd).get_value()
                weights = self.constraints[self.w_hh_name].np_constrain(weights) \
                    if self.constraints.get(self.w_hh_name, None) is not None else weights
        elif (isinstance(W_hh, theano.tensor.sharedvar.TensorSharedVariable)):
            weights = self.constraints[self.w_hh_name].np_constrain(W_hh.get_value()) \
                      if self.constraints.get(self.w_hh_name, None) is not None else W_hh.get_value()
        elif (isinstance(W_hh, np.ndarray)):
            weights = self.constraints[self.w_hh_name].np_constrain(W_hh) \
                      if self.constraints.get(self.w_hh_name, None) is not None else W_hh
        else:
            assert False, "Provided pretrained value for W_hh must be either a tensor shared variable or a " \
                          "numpy ndarray"

        if weights is not None:
            assert len(weights.shape) == 2 and weights.shape[0] == self.hidden_dim and \
                   weights.shape[1] == self.hidden_dim, "Provided pretrained W_hh weights are not of the " \
                                                        "appropriate shape!"
            if self.W_hh is not None:
                self.W_hh.set_value(weights)
            else:
                self.W_hh = theano.shared(value=weights, name=self.w_hh_name, strict=False)
        if self.W_hh is not None:
            self.trainable_params[self.w_hh_name] = self.W_hh

        biases = None

        if b_h is None:
            if init_b:
                biases = initializers.get_init_value(init_type='zero',name=self.b_h_name,
                                                     dim_tuple=self.b_h_dim_tuple).get_value()
                biases = self.constraints[self.b_h_name].np_constrain(biases) \
                    if self.constraints.get(self.b_h_name, None) is not None else biases
        elif isinstance(b_h, theano.tensor.sharedvar.TensorSharedVariable):
            biases = self.constraints[self.b_h_name].np_constrain(b_h.get_value()) \
                    if self.constraints.get(self.b_h_name, None) is not None else b_h.get_value()
        elif isinstance(b_h, np.ndarray):
            biases = self.constraints[self.b_h_name].np_constrain(b_h) \
                if self.constraints.get(self.b_h_name, None) is not None else b_h
        else:
            assert False, "Provided pretrained value for b_h must be a tensor shared variable or a numpy ndarray"

        if biases is not None:
            assert len(biases.shape) == 1 and biases.shape[0] == self.hidden_dim, "Provided pretrained biases are not" \
                                                                                  "of the appropriate shape!"
            if self.b_h is not None:
                self.b_h.set_value(biases)
            else:
                self.b_h = theano.shared(value=biases, name=self.b_h_name, strict=False)
        if self.b_h is not None and self.use_bias_h:
            self.trainable_params[self.b_h_name] = self.b_h

        self.trainable_param_names = self.trainable_params.keys()

    def get_test_output(self, input, init_hidden=None):
        return self.get_train_output(input=input, init_hidden=init_hidden)

    def get_train_output(self, input, init_hidden=None):

        eprint("Set theano default floatX to be float64 for RNNs to work! The scan functions involved mandate this!")

        def RNN_recurrence(transformed_curr_ip, prev_h):
            score = T.dot(prev_h, self.W_hh) + transformed_curr_ip
            # The transformed current ip already has the dot product with W_ih applied
            if(self.use_bias_h):
                 score += self.b_h
            return activations.get_activation(activ_type=self.activation,x=score,leak_slope=self.leak_slope,
                                             clip_threshold=self.clip_threshold)

        if self.with_batch:
            # In this case the input tensor must have dimension (batch_size, sequence_len, input_dim)
            assert input.ndim==3, "Input must have 3 dimensions: (batch_size, sequence_len, input_dim)"
            eprint("NEED TO IMPLEMENT LOSS FUNCTIONS THAT DEAL WITH A BATCHED SCENARIO!")
            # However, for the scan function, we need (sequence_len, batch_size, input_dim)
            input = input.dimshuffle(1,0,2)
            if (init_hidden is not None):
                assert init_hidden.ndim == 2, "Initial hidden state must have 2 dimensions: (batch_size, hidden_dim)"
            else:
                init_hidden = T.alloc(self.default_init_hid, input.shape[1], self.hidden_dim)
                # Generates tensor of size (batch_size, hidden dim) with all values set to 0
        else:
            assert(input.ndim==2), "Input must have 2 dimensions: (sequence_len, input_dim)"
            if(init_hidden is not None):
                assert init_hidden.ndim == 1, "Initial hidden state must have 1 dimension: (hidden_dim,)"
            else:
                init_hidden = self.default_init_hid

        h, _ = theano.scan(fn=RNN_recurrence,
                           sequences=[T.dot(input, self.W_ih)],
                           outputs_info=init_hidden,
                           n_steps=input.shape[0]
                           )
        if self.with_batch:
            h = h.dimshuffle(1,0,2) # Recovering the ordering of dimensions as in the original input

        return h

    def get_config(self):
        return{
                "name":self.layer_name,
                "hidden_dim":self.hidden_dim,
                "input_dim":self.input_dim,
                "hidden_dtype":self.hidden_dtype,
                "input_dtype":self.input_dtype,
                "activation":self.activation,
                "init_type":self.init_type,
                "with_batch":self.with_batch,
                "learning_rate":self.learning_rate,
                "leak_slope":self.leak_slope,
                "clip_threshold":self.clip_threshold,
                "use_bias_h":self.use_bias_h,
                "rnd_seed":self.rnd_seed,
                "rnd":self.rnd,
                "trainable":self.trainable
              }


class LSTM(RecurrentLayer):

    def __init__(self, name, hidden_dim, input_dim, input_dtype=Tconfig.floatX,
                 hidden_dtype=Tconfig.floatX, peephole_f=True, peephole_i=True, peephole_o=True,
                 activation='relu', gate_activation='sigmoid', init_type='glorot_uniform', with_batch=False,
                 learning_rate=0.01, leak_slope=0.01, clip_threshold=1.0,
                 weights_hf=None, weights_xf=None, weights_cf=None, biases_f=None,
                 weights_hi=None, weights_xi=None, weights_ci=None, biases_i=None,
                 weights_hu=None, weights_xu=None, biases_u=None,
                 weights_ho=None, weights_xo=None, weights_co=None, biases_o=None,

                 w_hf_regularizer=None, w_xf_regularizer=None, w_cf_regularizer=None, b_f_regularizer=None,
                 w_hu_regularizer=None, w_xu_regularizer=None, b_u_regularizer=None,
                 w_hi_regularizer=None, w_xi_regularizer=None, w_ci_regularizer=None, b_i_regularizer=None,
                 w_ho_regularizer=None, w_xo_regularizer=None, w_co_regularizer=None, b_o_regularizer=None,

                 w_hf_constraint=None, w_xf_constraint=None, w_cf_constraint=None, b_f_constraint=None,
                 w_hu_constraint=None, w_xu_constraint=None, b_u_constraint=None,
                 w_hi_constraint=None, w_xi_constraint=None, w_ci_constraint=None, b_i_constraint=None,
                 w_ho_constraint=None, w_xo_constraint=None, w_co_constraint=None, b_o_constraint=None,

                 use_biases=True, rnd_seed=None, trainable=True, **kwargs):
        super_args = {key: value for (key, value) in kwargs.items()}  # To avoid passing kwargs as a dictionary to super
        # constructor
        local_args = {key: value for (key, value) in locals().items()}  # This should catch all arguments input to this
        # constructor, but it includes self
        del local_args['self']
        del local_args['kwargs']
        super_args.update(local_args)
        del local_args
        super(LSTM, self).__init__(**super_args)
        self.context_dim = self.hidden_dim # In regular LSTMs, these are tied to be the same value. A more elaborator
        # model may want to decouple these
        self.peephole_f=peephole_f
        self.peephole_i=peephole_i
        self.peephole_o=peephole_o
        self.use_bias_h = use_biases
        self.gate_activation = gate_activation

        # Setting names for forget gate params

        self.w_hf_name=self.layer_name+"_w_hf"
        self.w_xf_name=self.layer_name+"_w_xf"
        self.w_cf_name=self.layer_name+"_w_cf"
        self.b_f_name=self.layer_name+"b_f"

        self.w_hf_dim = (self.hidden_dim, self.context_dim)
        self.w_xf_dim = (self.input_dim, self.context_dim)
        self.w_cf_dim = (self.context_dim, self.context_dim)
        self.b_f_dim = (self.context_dim,)

        self.w_hf = None
        self.w_xf = None
        self.w_cf = None
        self.b_f = None

        # Setting names for input gate weights
        self.w_hi_name=self.layer_name+"_w_hi"
        self.w_xi_name=self.layer_name+"_w_xi"
        self.w_ci_name=self.layer_name+"_w_ci"
        self.b_i_name=self.layer_name+"b_i"

        self.w_hi_dim = (self.hidden_dim, self.context_dim)
        self.w_xi_dim = (self.input_dim, self.context_dim)
        self.w_ci_dim = (self.context_dim, self.context_dim)
        self.b_i_dim = (self.context_dim,)

        self.w_hi = None
        self.w_xi = None
        self.w_ci = None
        self.b_i = None

        # Setting names for the weights used to determine the update
        self.w_hu_name=self.layer_name+"_w_hu"
        self.w_xu_name=self.layer_name+"_w_xu"
        self.b_u_name=self.layer_name+"b_u"

        self.w_hu_dim = (self.hidden_dim, self.context_dim)
        self.w_xu_dim = (self.input_dim, self.context_dim)
        self.b_u_dim = (self.context_dim,)

        self.w_hu = None
        self.w_xu = None
        self.b_u = None

        # Setting names for weights of output gate weights
        self.w_ho_name=self.layer_name+"_w_ho"
        self.w_xo_name=self.layer_name+"_w_xo"
        self.w_co_name=self.layer_name+"_w_co"
        self.b_o_name=self.layer_name+"_b_o"

        self.w_ho_dim = (self.hidden_dim, self.context_dim)
        self.w_xo_dim = (self.input_dim, self.context_dim)
        self.w_co_dim = (self.context_dim, self.context_dim)
        self.b_o_dim = (self.context_dim,)

        self.w_ho = None
        self.w_xo = None
        self.w_co = None
        self.b_o = None

        assert not (any(biases is not None for biases in [biases_f, biases_i, biases_o, biases_u]) and not use_biases),\
            "Pretrained bias weights provided to layer that doesn't use biases!"

        assert not (any(c_weight is not None and not peephole for (c_weight,peephole) in
                        zip([weights_cf, weights_ci, weights_co],[peephole_f, peephole_i, peephole_o]))), \
            "Pretrained weights for peephole connection provided to layer that doesn't use peepholes!"

        weights_hf = np.array(weights_hf) if weights_hf is not None and isinstance(weights_hf, list) else weights_hf
        weights_xf = np.array(weights_xf) if weights_xf is not None and isinstance(weights_xf, list) else weights_xf
        weights_cf = np.array(weights_cf) if weights_cf is not None and isinstance(weights_cf, list) else weights_cf
        biases_f = np.array(biases_f) if biases_f is not None and isinstance(biases_f, list) else biases_f

        weights_hu = np.array(weights_hu) if weights_hu is not None and isinstance(weights_hu, list) else weights_hu
        weights_xu = np.array(weights_xu) if weights_xu is not None and isinstance(weights_xu, list) else weights_xu
        biases_u = np.array(biases_u) if biases_u is not None and isinstance(biases_u, list) else biases_u

        weights_hi = np.array(weights_hi) if weights_hi is not None and isinstance(weights_hi, list) else weights_hi
        weights_xi = np.array(weights_xi) if weights_xi is not None and isinstance(weights_xi, list) else weights_xi
        weights_ci = np.array(weights_ci) if weights_ci is not None and isinstance(weights_ci, list) else weights_ci
        biases_i = np.array(biases_i) if biases_i is not None and isinstance(biases_i, list) else biases_i

        weights_ho = np.array(weights_ho) if weights_ho is not None and isinstance(weights_ho, list) else weights_ho
        weights_xo = np.array(weights_xo) if weights_xo is not None and isinstance(weights_xo, list) else weights_xo
        weights_co = np.array(weights_co) if weights_co is not None and isinstance(weights_co, list) else weights_co
        biases_o = np.array(biases_o) if biases_o is not None and isinstance(biases_o, list) else biases_o


        if any(param is None for param in [weights_hf, weights_xf, weights_cf, biases_f,
                                           weights_hu, weights_xu, biases_u,
                                           weights_hi, weights_xi, weights_ci, biases_i,
                                           weights_ho, weights_xo, weights_co, biases_o]):
            if self.rnd_seed is None:
                self.rnd = np.random.RandomState()
            else:
                self.rnd = np.random.RandomState(self.rnd_seed)

        for (regul, name) in zip([w_hf_regularizer, w_xf_regularizer, w_cf_regularizer, b_f_regularizer,
                                   w_hu_regularizer, w_xu_regularizer, b_u_regularizer,
                                   w_hi_regularizer, w_xi_regularizer, w_ci_regularizer, b_i_regularizer,
                                   w_ho_regularizer, w_xo_regularizer, w_co_regularizer, b_o_regularizer],
                                  [self.w_hf_name, self.w_xf_name, self.w_cf_name, self.b_f_name,
                                   self.w_hu_name, self.w_xu_name, self.b_u_name,
                                   self.w_hi_name, self.w_xi_name, self.w_ci_name, self.b_i_name,
                                   self.w_ho_name, self.w_xo_name, self.w_co_name, self.b_o_name]):
            if (regul is not None):
                assert isinstance(regul, regularizers.Regularizer), "Specified " + name + " regularizer must be an " \
                                                                                        "instance of the Regularizer " \
                                                                                        "class!"
                self.regularizers[name] = regul

        for (constr, name) in zip([w_hf_constraint, w_xf_constraint, w_cf_constraint, b_f_constraint,
                                  w_hu_constraint, w_xu_constraint, b_u_constraint,
                                  w_hi_constraint, w_xi_constraint, w_ci_constraint, b_i_constraint,
                                  w_ho_constraint, w_xo_constraint, w_co_constraint, b_o_constraint],
                                 [self.w_hf_name, self.w_xf_name, self.w_cf_name, self.b_f_name,
                                  self.w_hu_name, self.w_xu_name, self.b_u_name,
                                  self.w_hi_name, self.w_xi_name, self.w_ci_name, self.b_i_name,
                                  self.w_ho_name, self.w_xo_name, self.w_co_name, self.b_o_name]):

            if (constr is not None):
                assert isinstance(constr, constraints.Constraint), "Specified " + name + " constraint must be an " \
                                                                                          "instance of the Constraint " \
                                                                                          "class!"
                self.constraints[name] = constr

        # Actually initializing the LSTM parameters

        self.set_params(weights_hf=weights_hf, weights_xf=weights_xf, weights_cf=weights_cf, biases_f=biases_f,
                   weights_hi=weights_hi, weights_xi=weights_xi, weights_ci=weights_ci, biases_i=biases_i,
                   weights_hu=weights_hu, weights_xu=weights_xu, biases_u=biases_u,
                   weights_ho=weights_ho, weights_xo=weights_xo, weights_co=weights_co, biases_o=biases_o,
                   init_whx=True, init_wc=True,
                   init_b=self.use_bias_h)

        default_init_hid_dim_tup = (self.hidden_dim,)
        default_init_context_dim_tup = (self.context_dim, )
        self.default_init_hid = initializers.get_init_value(init_type='zero',
                                                            name=self.layer_name + "_default_hid_init",
                                                            dim_tuple=default_init_hid_dim_tup)

        self.default_context_hid = initializers.get_init_value(init_type='zero',
                                                            name=self.layer_name + "_default_hid_init",
                                                            dim_tuple=default_init_context_dim_tup)

    def set_w_h(self, weights_hf=None, weights_hi=None, weights_hu=None, weights_ho=None, init_w=False):
        if weights_hf is None:
            if init_w:
                self.w_hf = initializers.get_init_value(init_type=self.init_type, name=self.w_hf_name,
                                                        dim_tuple=self.w_hf_dim,
                                                        constraint=self.constraints.get(self.w_hf_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_hf, np.ndarray):
            self.w_hf = initializers.get_init_value(init_type=self.init_type, name=self.w_hf_name,
                                                    dim_tuple=self.w_hf_dim,
                                                    constraint=self.constraints.get(self.w_hf_name, None),
                                                    weights=weights_hf)
        elif isinstance(weights_hf, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_hf = initializers.get_init_value(init_type=self.init_type, name=self.w_hf_name,
                                                    dim_tuple=self.w_hf_dim, weights=weights_hf.get_value(),
                                                    constraint=self.constraints.get(self.w_hf_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_hf is not None:
            self.trainable_params[self.w_hf_name] = self.w_hf

        if weights_hi is None:
            if init_w:
                self.w_hi = initializers.get_init_value(init_type=self.init_type, name=self.w_hi_name,
                                                        dim_tuple=self.w_hi_dim,
                                                        constraint=self.constraints.get(self.w_hi_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_hi, np.ndarray):
            self.w_hi = initializers.get_init_value(init_type=self.init_type, name=self.w_hi_name,
                                                    dim_tuple=self.w_hi_dim,
                                                    constraint=self.constraints.get(self.w_hi_name, None),
                                                    weights=weights_hi)
        elif isinstance(weights_hi, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_hi = initializers.get_init_value(init_type=self.init_type, name=self.w_hi_name,
                                                    dim_tuple=self.w_hi_dim,weights=weights_hi.get_value(),
                                                    constraint=self.constraints.get(self.w_hi_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_hi is not None:
            self.trainable_params[self.w_hi_name] = self.w_hi

        if weights_hu is None:
            if init_w:
                self.w_hu = initializers.get_init_value(init_type=self.init_type, name=self.w_hu_name,
                                                        dim_tuple=self.w_hu_dim,
                                                        constraint=self.constraints.get(self.w_hu_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_hu, np.ndarray):
            self.w_hu = initializers.get_init_value(init_type=self.init_type, name=self.w_hu_name,
                                                    dim_tuple=self.w_hu_dim,
                                                    constraint=self.constraints.get(self.w_hu_name, None),
                                                    weights=weights_hu)
        elif isinstance(weights_hu, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_hu = initializers.get_init_value(init_type=self.init_type, name=self.w_hu_name,
                                                    dim_tuple=self.w_hu_dim, weights=weights_hu.get_value(),
                                                    constraint=self.constraints.get(self.w_hu_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_hu is not None:
            self.trainable_params[self.w_hu_name] = self.w_hu

        if weights_ho is None:
            if init_w:
                self.w_ho = initializers.get_init_value(init_type=self.init_type, name=self.w_ho_name,
                                                        dim_tuple=self.w_ho_dim,
                                                        constraint=self.constraints.get(self.w_ho_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_ho, np.ndarray):
            self.w_ho = initializers.get_init_value(init_type=self.init_type, name=self.w_ho_name,
                                                    dim_tuple=self.w_ho_dim,
                                                    constraint=self.constraints.get(self.w_ho_name, None),
                                                    weights=weights_ho)
        elif isinstance(weights_ho, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_ho = initializers.get_init_value(init_type=self.init_type, name=self.w_ho_name,
                                                    dim_tuple=self.w_ho_dim, weights=weights_ho.get_value(),
                                                    constraint=self.constraints.get(self.w_ho_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_ho is not None:
            self.trainable_params[self.w_ho_name] = self.w_ho

    def set_w_x(self, weights_xf=None, weights_xi=None, weights_xu=None, weights_xo=None, init_w=False):
        if weights_xf is None:
            if init_w:
                self.w_xf = initializers.get_init_value(init_type=self.init_type, name=self.w_xf_name,
                                                        dim_tuple=self.w_xf_dim,
                                                        constraint=self.constraints.get(self.w_xf_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_xf, np.ndarray):
            self.w_xf = initializers.get_init_value(init_type=self.init_type, name=self.w_xf_name,
                                                    dim_tuple=self.w_xf_dim,
                                                    constraint=self.constraints.get(self.w_xf_name, None),
                                                    weights=weights_xf)
        elif isinstance(weights_xf, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_xf = initializers.get_init_value(init_type=self.init_type, name=self.w_xf_name,
                                                    dim_tuple=self.w_xf_dim, weights=weights_xf.get_value(),
                                                    constraint=self.constraints.get(self.w_xf_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_xf is not None:
            self.trainable_params[self.w_xf_name] = self.w_xf

        if weights_xi is None:
            if init_w:
                self.w_xi = initializers.get_init_value(init_type=self.init_type, name=self.w_xi_name,
                                                        dim_tuple=self.w_xi_dim,
                                                        constraint=self.constraints.get(self.w_xi_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_xi, np.ndarray):
            self.w_xi = initializers.get_init_value(init_type=self.init_type, name=self.w_xi_name,
                                                    dim_tuple=self.w_xi_dim,
                                                    constraint=self.constraints.get(self.w_xi_name, None),
                                                    weights=weights_xi)
        elif isinstance(weights_xi, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_xi = initializers.get_init_value(init_type=self.init_type, name=self.w_xi_name,
                                                    dim_tuple=self.w_xi_dim, weights=weights_xi.get_value(),
                                                    constraint=self.constraints.get(self.w_xi_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_xi is not None:
            self.trainable_params[self.w_xi_name] = self.w_xi

        if weights_xu is None:
            if init_w:
                self.w_xu = initializers.get_init_value(init_type=self.init_type, name=self.w_xu_name,
                                                        dim_tuple=self.w_xu_dim,
                                                        constraint=self.constraints.get(self.w_xu_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_xu, np.ndarray):
            self.w_xu = initializers.get_init_value(init_type=self.init_type, name=self.w_xu_name,
                                                    dim_tuple=self.w_xu_dim,
                                                    constraint=self.constraints.get(self.w_xu_name, None),
                                                    weights=weights_xu)
        elif isinstance(weights_xu, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_xu = initializers.get_init_value(init_type=self.init_type, name=self.w_xu_name,
                                                    dim_tuple=self.w_xu_dim, weights=weights_xu.get_value(),
                                                    constraint=self.constraints.get(self.w_xu_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_xu is not None:
            self.trainable_params[self.w_xu_name] = self.w_xu

        if weights_xo is None:
            if init_w:
                self.w_xo = initializers.get_init_value(init_type=self.init_type, name=self.w_xo_name,
                                                        dim_tuple=self.w_xo_dim,
                                                        constraint=self.constraints.get(self.w_xo_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_xo, np.ndarray):
            self.w_xo = initializers.get_init_value(init_type=self.init_type, name=self.w_xo_name,
                                                    dim_tuple=self.w_xo_dim,
                                                    constraint=self.constraints.get(self.w_xo_name, None),
                                                    weights=weights_xo)
        elif isinstance(weights_xo, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_xo = initializers.get_init_value(init_type=self.init_type, name=self.w_xo_name,
                                                    dim_tuple=self.w_xo_dim, weights=weights_xo.get_value(),
                                                    constraint=self.constraints.get(self.w_xo_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_xo is not None:
            self.trainable_params[self.w_xo_name] = self.w_xo

    def set_w_c(self, weights_cf=None, weights_ci=None, weights_co=None, init_w=False):
        if weights_cf is not None:
            assert (self.peephole_f), "Can't initialize context weights forget gate without peephole allowed!"
        if weights_ci is not None:
            assert (self.peephole_i), "Can't initialize context weights input gate without peephole allowed!"
        if weights_co is not None:
            assert (self.peephole_o), "Can't initialize context weights output gate without peephole allowed!"


        if weights_cf is None:
            if init_w and self.peephole_f:
                self.w_cf = initializers.get_init_value(init_type=self.init_type, name=self.w_cf_name,
                                                        dim_tuple=self.w_cf_dim,
                                                        constraint=self.constraints.get(self.w_cf_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_cf, np.ndarray):
            self.w_cf = initializers.get_init_value(init_type=self.init_type, name=self.w_cf_name,
                                                    dim_tuple=self.w_cf_dim,
                                                    constraint=self.constraints.get(self.w_cf_name, None),
                                                    weights=weights_cf)
        elif isinstance(weights_cf, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_cf = initializers.get_init_value(init_type=self.init_type, name=self.w_cf_name,
                                                    dim_tuple=self.w_cf_dim, weights=weights_cf.get_value(),
                                                    constraint=self.constraints.get(self.w_cf_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_cf is not None:
            self.trainable_params[self.w_cf_name] = self.w_cf


        if weights_ci is None:
            if init_w and self.peephole_i:
                self.w_ci = initializers.get_init_value(init_type=self.init_type, name=self.w_ci_name,
                                                        dim_tuple=self.w_ci_dim,
                                                        constraint=self.constraints.get(self.w_ci_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_ci, np.ndarray):
            self.w_ci = initializers.get_init_value(init_type=self.init_type, name=self.w_ci_name,
                                                    dim_tuple=self.w_ci_dim,
                                                    constraint=self.constraints.get(self.w_ci_name, None),
                                                    weights=weights_ci)
        elif isinstance(weights_ci, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_ci = initializers.get_init_value(init_type=self.init_type, name=self.w_ci_name,
                                                    dim_tuple=self.w_ci_dim, weights=weights_ci.get_value(),
                                                    constraint=self.constraints.get(self.w_ci_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_ci is not None:
            self.trainable_params[self.w_ci_name] = self.w_ci

        if weights_co is None:
            if init_w and self.peephole_o:
                self.w_co = initializers.get_init_value(init_type=self.init_type, name=self.w_co_name,
                                                        dim_tuple=self.w_co_dim,
                                                        constraint=self.constraints.get(self.w_co_name, None),
                                                        rnd=self.rnd)
        elif isinstance(weights_co, np.ndarray):
            self.w_co = initializers.get_init_value(init_type=self.init_type, name=self.w_co_name,
                                                    dim_tuple=self.w_co_dim,
                                                    constraint=self.constraints.get(self.w_co_name, None),
                                                    weights=weights_co)
        elif isinstance(weights_co, theano.tensor.sharedvar.TensorSharedVariable):
            self.w_co = initializers.get_init_value(init_type=self.init_type, name=self.w_co_name,
                                                    dim_tuple=self.w_co_dim, weights=weights_co.get_value(),
                                                    constraint=self.constraints.get(self.w_co_name, None))
            # Note that this creates a copy of the shared variables value contents. This prevents unforeseen problems
            # like different layers using the same shared variable.
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable"

        if self.w_co is not None:
            self.trainable_params[self.w_co_name] = self.w_co

    def set_biases(self, bias_f=None, bias_i=None, bias_u=None, bias_o=None, init_b=None):
        if bias_f is None:
            if init_b:
                self.b_f = initializers.get_init_value(init_type='zero',name=self.b_f_name,dim_tuple=self.b_f_dim,
                                                       constraint=self.constraints.get(self.b_f_name, None))
        elif isinstance(bias_f, np.ndarray):
            self.b_f = initializers.get_init_value(init_type='zero', name=self.b_f_name,
                                                   dim_tuple=self.b_f_dim,weights=bias_f,
                                                   constraint=self.constraints.get(self.b_f_name, None))
        elif isinstance(bias_f, theano.tensor.sharedvar.TensorSharedVariable):
            self.b_f = initializers.get_init_value(init_type='zero', name=self.b_f_name, dim_tuple=self.b_f_dim,
                                                   weights=bias_f.get_value(),
                                                   constraint=self.constraints.get(self.b_f_name, None))
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable!"

        if self.b_f is not None:
            self.trainable_params[self.b_f_name] = self.b_f

        if bias_i is None:
            if init_b:
                self.b_i = initializers.get_init_value(init_type='zero', name=self.b_i_name, dim_tuple=self.b_i_dim,
                                                       constraint=self.constraints.get(self.b_i_name, None))
        elif isinstance(bias_i, np.ndarray):
            self.b_i = initializers.get_init_value(init_type='zero', name=self.b_i_name,
                                                   dim_tuple=self.b_i_dim, weights=bias_i,
                                                   constraint=self.constraints.get(self.b_i_name, None))
        elif isinstance(bias_i, theano.tensor.sharedvar.TensorSharedVariable):
            self.b_i = initializers.get_init_value(init_type='zero', name=self.b_i_name, dim_tuple=self.b_i_dim,
                                                   weights=bias_i.get_value(),
                                                   constraint=self.constraints.get(self.b_i_name, None))
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable!"

        if self.b_i is not None:
            self.trainable_params[self.b_i_name] = self.b_i

        if bias_o is None:
            if init_b:
                self.b_o = initializers.get_init_value(init_type='zero', name=self.b_o_name, dim_tuple=self.b_o_dim,
                                                       constraint=self.constraints.get(self.b_o_name, None))
        elif isinstance(bias_o, np.ndarray):
            self.b_o = initializers.get_init_value(init_type='zero', name=self.b_o_name,
                                                   dim_tuple=self.b_o_dim, weights=bias_o,
                                                   constraint=self.constraints.get(self.b_o_name, None))
        elif isinstance(bias_o, theano.tensor.sharedvar.TensorSharedVariable):
            self.b_o = initializers.get_init_value(init_type='zero', name=self.b_o_name, dim_tuple=self.b_o_dim,
                                                   weights=bias_o.get_value(),
                                                   constraint=self.constraints.get(self.b_o_name, None))
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable!"

        if self.b_o is not None:
            self.trainable_params[self.b_o_name] = self.b_o

        if bias_u is None:
            if init_b:
                self.b_u = initializers.get_init_value(init_type='zero', name=self.b_u_name, dim_tuple=self.b_u_dim,
                                                       constraint=self.constraints.get(self.b_u_name, None))
        elif isinstance(bias_u, np.ndarray):
            self.b_u = initializers.get_init_value(init_type='zero', name=self.b_u_name,
                                                   dim_tuple=self.b_u_dim, weights=bias_u,
                                                   constraint=self.constraints.get(self.b_u_name, None))
        elif isinstance(bias_u, theano.tensor.sharedvar.TensorSharedVariable):
            self.b_u = initializers.get_init_value(init_type='zero', name=self.b_u_name, dim_tuple=self.b_u_dim,
                                                   weights=bias_u.get_value(),
                                                   constraint=self.constraints.get(self.b_u_name, None))
        else:
            assert False, "Can only initialize LSTM params with a numpy ndarray or theano shared tensor variable!"

        if self.b_u is not None:
            self.trainable_params[self.b_u_name] = self.b_u

    def set_params(self, weights_hf=None, weights_xf=None, weights_cf=None, biases_f=None,
                   weights_hi=None, weights_xi=None, weights_ci=None, biases_i=None,
                   weights_hu=None, weights_xu=None, biases_u=None,
                   weights_ho=None, weights_xo=None, weights_co=None, biases_o=None,
                   init_whx=False, init_wc=False,
                   init_b=False):
        self.set_w_h(weights_hf=weights_hf,weights_hi=weights_hi,weights_hu=weights_hu,weights_ho=weights_ho,
                     init_w=init_whx)
        self.set_w_x(weights_xf=weights_xf, weights_xi=weights_xi, weights_xu=weights_xu, weights_xo=weights_xo,
                     init_w=init_whx)
        self.set_w_c(weights_cf=weights_cf, weights_ci=weights_ci, weights_co=weights_co,
                     init_w=init_wc)
        self.set_biases(bias_f=biases_f, bias_i=biases_i, bias_u=biases_u, bias_o=biases_o, init_b=init_b)
        self.trainable_param_names = self.trainable_params.keys()

    def get_config(self):
        return {
            "name"          : self.layer_name,
            "hidden_dim"    : self.hidden_dim,
            "input_dim"     : self.input_dim,
            "context_dim"   : self.context_dim,
            "peephole_f"    : self.peephole_f,
            "peephole_i"    : self.peephole_i,
            "peephole_o"    : self.peephole_o,
            "hidden_dtype"  : self.hidden_dtype,
            "input_dtype"   : self.input_dtype,
            "activation"    : self.activation,
            "init_type"     : self.init_type,
            "with_batch"    : self.with_batch,
            "learning_rate" : self.learning_rate,
            "leak_slope"    : self.leak_slope,
            "clip_threshold": self.clip_threshold,
            "use_bias_h"    : self.use_bias_h,
            "rnd_seed"      : self.rnd_seed,
            "rnd"           : self.rnd,
            "trainable"     : self.trainable
        }

    def link(self, input, init_hidden=None, init_context=None, is_train=True):
        assert isinstance(input, theano.tensor.TensorVariable), "Input tensor must be a theano tensor " \
                                                                "variable (theano.tensor.TensorVariable)"

        assert init_hidden is None or isinstance(init_hidden,
                                                 theano.tensor.TensorVariable), "Init hidden must be a theano tensor " \
                                                                                "variable (theano.tensor.TensorVariable)"

        assert init_context is None or isinstance(init_context,
                                                 theano.tensor.TensorVariable), "Init context must be a theano tensor "\
                                                                                "variable (" \
                                                                                "theano.tensor.TensorVariable)"

        if is_train:
            return (self.get_train_output(input, init_hidden, init_context))
        else:
            return (self.get_test_output(input, init_hidden, init_context))

    def get_test_output(self, input, init_hidden, init_context):
        return self.get_train_output(input, init_hidden, init_context)

    def get_train_output(self, input, init_hidden, init_context):

        # Note that for an LSTM, only the hidden state can be pre-seeded. The context will have to be set to 0s
        # initially. If the context is also seeded, then the output LSTM essentially becomes an extension of the
        # previous LSTM

        def lstm_recurrence(x_t, h_tm1, c_tm1):
            f_t = T.dot(x_t, self.w_xf) + T.dot(h_tm1, self.w_hf)
            if(self.peephole_f):
                f_t += T.dot(c_tm1, self.w_cf)

            i_t = T.dot(x_t, self.w_xi) + T.dot(h_tm1, self.w_hi)
            if(self.peephole_i):
                i_t += T.dot(c_tm1, self.w_ci)

            o_t = T.dot(x_t, self.w_xo) + T.dot(h_tm1, self.w_ho)
            if(self.peephole_o):
                o_t += T.dot(c_tm1, self.w_co)

            u_t = T.dot(x_t, self.w_xu) + T.dot(h_tm1, self.w_hu)

            if(self.use_bias_h):
                f_t += self.b_f
                i_t += self.b_i
                o_t += self.b_o
                u_t += self.b_u

            f_t = activations.get_activation(self.gate_activation, f_t)
            i_t = activations.get_activation(self.gate_activation, i_t)
            o_t = activations.get_activation(self.gate_activation, o_t)
            u_t = activations.get_activation(self.activation, u_t)

            u_t = (c_tm1 * f_t) + (u_t * i_t)

            h = activations.get_activation(self.activation, u_t) * o_t
            return [h, u_t]

        if self.with_batch:
            assert input.ndim==3, "Input must have 3 dimensions: (batch_size, sequence_len, input_dim)"
            eprint("NEED TO IMPLEMENT LOSS FUNCTIONS THAT DEAL WITH A BATCHED SCENARIO!")
            input = input.dimshuffle(1,0,2) # Make the sequence length the leading dimension

            if (init_hidden is not None):
                assert init_hidden.ndim == 2, "Initial hidden state must have 2 dimensions: (batch_size, hidden_dim)"
            else:
                init_hidden = T.alloc(self.default_init_hid, input.shape[1], self.hidden_dim)
                # Generates tensor of size (batch_size, hidden dim) with all values set to 0
            if (init_context is not None):
                assert init_context.ndim == 2, "Initial context state must have 2 dimensions: (batch_size, context_dim)"
            else:
                init_context = T.alloc(self.default_context_hid, input.shape[1], self.context_dim)
                # Generates tensor of size (batch_size, hidden dim) with all values set to 0
        else:
            assert (input.ndim == 2), "Input must have 2 dimensions: (sequence_len, input_dim)"
            if (init_hidden is not None):
                assert init_hidden.ndim == 1, "Initial hidden state must have 1 dimension: (hidden_dim,)"
            else:
                init_hidden = self.default_init_hid

            if(init_context is not None):
                assert init_context.ndim == 1, "Initial context state must have 1 dimension: (context_dim,)"
            else:
                init_context = self.default_context_hid

        [h, c], _ = theano.scan(fn=lstm_recurrence,
                           sequences=[input],
                           outputs_info=[init_hidden, init_context],
                           n_steps=input.shape[0]
                           )

        if self.with_batch:
            h = h.dimshuffle(1,0,2)
            c = c.dimshuffle(1,0,2)

        return (h, c)