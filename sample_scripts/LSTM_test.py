import sys
import theano
import theano.tensor as T
import numpy as np

sys.path.append("/home/akashb/Desktop/Acads/Summer/Leibnetz/Leibnetz")
sys.path.append("/home/akashb/Desktop/Acads/Summer/Leibnetz/Leibnetz/datasets")
from layers import *
from recurrent_layers import *
from model import *
from optimizers import *
from dataset_utils import *
from regularizers import *
from constraints import *
from model import *

hidden_dim = 15
context_dim = 15

# context_dim must be same as hidden_dim

ip_dim = 100

use_batch = True

# With batching

ip_tensor = T.dtensor3("sgd_ip") if use_batch else T.dmatrix("sgd_ip")

id_xc = np.eye(ip_dim, context_dim)
id_hh = np.eye(hidden_dim, hidden_dim)
id_ch = np.eye(context_dim, hidden_dim)
id_cc = np.eye(context_dim, context_dim)
id_hc = np.eye(hidden_dim, context_dim)
bvec = np.ones(shape=(context_dim))

# rnn = RNN(name="sgd_rnn", hidden_dim=hidden_dim, input_dim=ip_dim, activation = 'linear', init_type = 'one',
#           weights_ih = 2*identity_mat, weights_hh = 3*identity_mat,biases_h = np.ones(shape=(hidden_dim,)),
#           W_ih_regularizer = None, W_hh_regularizer = None,
#           W_ih_constraint = None, W_hh_constraint = None, b_h_regularizer = None, b_h_constraint = None,
#           use_bias_h = True, rnd_seed = None, trainable = True, with_batch=True)

lstm = LSTM(name="batch_lstm", hidden_dim=hidden_dim, input_dim=ip_dim,
            peephole_f=True, peephole_i=True,
            peephole_o=True, gate_activation='sigmoid',
            activation='tanh', init_type='glorot_uniform', with_batch=use_batch,
            leak_slope=0.01, clip_threshold=5.0,
            weights_hf=id_hc, weights_xf=id_xc, weights_cf=id_cc, biases_f=bvec,
            weights_hi=id_hc, weights_xi=id_xc, weights_ci=id_cc, biases_i=bvec,
            weights_hu=id_hc, weights_xu=id_xc, biases_u=bvec,
            weights_ho=id_hc, weights_xo=id_xc, weights_co=id_cc, biases_o=bvec,

            w_hf_regularizer=None, w_xf_regularizer=None, w_cf_regularizer=None, b_f_regularizer=None,
            w_hu_regularizer=None, w_xu_regularizer=None, b_u_regularizer=None,
            w_hi_regularizer=None, w_xi_regularizer=None, w_ci_regularizer=None, b_i_regularizer=None,
            w_ho_regularizer=None, w_xo_regularizer=None, w_co_regularizer=None, b_o_regularizer=None,

            w_hf_constraint=None, w_xf_constraint=None, w_cf_constraint=None, b_f_constraint=None,
            w_hu_constraint=None, w_xu_constraint=None, b_u_constraint=None,
            w_hi_constraint=None, w_xi_constraint=None, w_ci_constraint=None, b_i_constraint=None,
            w_ho_constraint=None, w_xo_constraint=None, w_co_constraint=None, b_o_constraint=None,
            w_ch_constraint=None, b_h_constraint=None,

            use_biases=True, rnd_seed=None, trainable=True)

# test_op = rnn.link(ip_tensor, init_hidden=init_state, is_train=False)
[test_op, test_c ] = lstm.link(ip_tensor, init_hidden=None, init_context=None, is_train=False)

# get_op = theano.function(inputs=[ip_tensor, init_state],outputs=[test_op])
get_op = theano.function(inputs=[ip_tensor],outputs=[test_op])

lstm2 = LSTM(name="batch_lstm2", hidden_dim=hidden_dim, input_dim=ip_dim,
             peephole_f=True, peephole_i=True,
             peephole_o=True, gate_activation='sigmoid',
             activation='tanh', init_type='glorot_uniform', with_batch=use_batch,
             leak_slope=0.01, clip_threshold=5.0,
             weights_hf=id_hc, weights_xf=id_xc, weights_cf=id_cc, biases_f=bvec,
             weights_hi=id_hc, weights_xi=id_xc, weights_ci=id_cc, biases_i=bvec,
             weights_hu=id_hc, weights_xu=id_xc, biases_u=bvec,
             weights_ho=id_hc, weights_xo=id_xc, weights_co=id_cc, biases_o=bvec,

             w_hf_regularizer=None, w_xf_regularizer=None, w_cf_regularizer=None, b_f_regularizer=None,
             w_hu_regularizer=None, w_xu_regularizer=None, b_u_regularizer=None,
             w_hi_regularizer=None, w_xi_regularizer=None, w_ci_regularizer=None, b_i_regularizer=None,
             w_ho_regularizer=None, w_xo_regularizer=None, w_co_regularizer=None, b_o_regularizer=None,
             w_ch_regularizer=None, b_h_regularizer=None,

             w_hf_constraint=None, w_xf_constraint=None, w_cf_constraint=None, b_f_constraint=None,
             w_hu_constraint=None, w_xu_constraint=None, b_u_constraint=None,
             w_hi_constraint=None, w_xi_constraint=None, w_ci_constraint=None, b_i_constraint=None,
             w_ho_constraint=None, w_xo_constraint=None, w_co_constraint=None, b_o_constraint=None,
             w_ch_constraint=None, b_h_constraint=None,

             use_biases=True, rnd_seed=None, trainable=True)

# [test_op2, test_c2 ] = lstm2.link(ip_tensor, init_hidden=test_op[:, -1, :], init_context=test_c[:,-1,:], is_train=False)
if use_batch:
    # Seeding both the context and hidden states
    # [test_op2, test_c2 ] = lstm2.link(ip_tensor, init_hidden=test_op[:, -1, :], init_context=test_c[:, -1, :], is_train=False)
    # Seeding only the hidden state
    [test_op2, test_c2] = lstm2.link(ip_tensor, init_hidden=test_op[:, -1, :],
                                     is_train=False)
else:
    # Seeding both the context and hidden states
    # [test_op2, test_c2] = lstm2.link(ip_tensor, init_hidden=test_op[-1, :], init_context=test_c[-1, :],
    #                                  is_train=False)
    # Seeding only the hidden state
    [test_op2, test_c2] = lstm2.link(ip_tensor, init_hidden=test_op[-1, :],
                                     is_train=False)

# Extracting the last hidden state output for each batch sample

get_op2 = theano.function(inputs=[ip_tensor], outputs=[test_op2])

test_ip = np.ones(shape=(5,ip_dim),dtype=np.float64)

if use_batch:
    test_ip_batch = [test_ip, 2*test_ip]
else:
    test_ip_batch = test_ip;

print("\n\nBatch test ip1")
test_op1 = get_op(test_ip_batch)
print(len(test_op1))
print(test_op1[:][:][-1])
for batch in test_op1:
    print(batch)
    raw_input("Enter to continue")

print("\n\nBatch test ip2")
print(get_op2(test_ip_batch))

# Try with relu

# FIX THE intiialization routine and check that constraints are correctly enforced
# Also try RNN without the bias terms
# Try with different dimensions for hidden, input and context