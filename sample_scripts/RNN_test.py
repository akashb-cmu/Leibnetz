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

hidden_dim = 100
ip_dim = 100

# Without batching
#
# ip_tensor = T.dmatrix("sgd_ip");
# init_state = T.dvector("init_hid")
#
# identity_mat = np.eye(ip_dim, hidden_dim)
#
# rnn = RNN(name="sgd_rnn", hidden_dim=hidden_dim, input_dim=ip_dim, activation = 'linear', init_type = 'one',
#           weights_ih = 2*identity_mat, weights_hh = 3*identity_mat,biases_h = np.ones(shape=(hidden_dim,)),
#           W_ih_regularizer = None, W_hh_regularizer = None,
#           W_ih_constraint = None, W_hh_constraint = None, b_h_regularizer = None, b_h_constraint = None,
#           use_bias_h = True, rnd_seed = None, trainable = True)
# # test_op = rnn.link(ip_tensor, init_hidden=init_state, is_train=False)
# test_op = rnn.link(ip_tensor, init_hidden=None, is_train=False)
#
# # get_op = theano.function(inputs=[ip_tensor, init_state],outputs=[test_op])
# get_op = theano.function(inputs=[ip_tensor],outputs=[test_op])
#
# get_ip_transform = theano.function(inputs=[ip_tensor], outputs=[T.dot(ip_tensor, rnn.W_ih)])
#
# rnn2 = RNN(name="sgd_rnn_2", hidden_dim=hidden_dim, input_dim=ip_dim, activation = 'linear', init_type = 'one',
#            weights_ih = identity_mat, weights_hh = identity_mat,biases_h = np.ones(shape=(hidden_dim,)),
#            W_ih_regularizer = None, W_hh_regularizer = None,
#            W_ih_constraint = None, W_hh_constraint = None, b_h_regularizer = None, b_h_constraint = None,
#            use_bias_h = True, rnd_seed = None, trainable = True)
#
# test_op2 = rnn2.link(ip_tensor, init_hidden=test_op[-1], is_train=False)
#
# get_op2 = theano.function(inputs=[ip_tensor], outputs=[test_op2])
#
#
# test_ip = np.ones(shape=(5,ip_dim),dtype=np.float64)
# # test_init_hid = np.ones(shape=(hidden_dim,), dtype=np.float64)
# # test_init_hid = np.zeros(shape=(hidden_dim,), dtype=np.float64)
# test_init_hid=None
# # print(get_op(test_ip, test_init_hid))
#
# # print("Input transform only:")
# # print(get_ip_transform(test_ip))
# #
# # raw_input("Enter to continue")
#
# print("\n\ntest_op1:\n")
# print(get_op(test_ip))
#
# print("\n\ntest_op2:\n")
# print(get_op2(test_ip))

# With batching

ip_tensor = T.dtensor3("sgd_ip");

identity_mat = np.eye(ip_dim, hidden_dim)

# rnn = RNN(name="sgd_rnn", hidden_dim=hidden_dim, input_dim=ip_dim, activation = 'linear', init_type = 'one',
#           weights_ih = 2*identity_mat, weights_hh = 3*identity_mat,biases_h = np.ones(shape=(hidden_dim,)),
#           W_ih_regularizer = None, W_hh_regularizer = None,
#           W_ih_constraint = None, W_hh_constraint = None, b_h_regularizer = None, b_h_constraint = None,
#           use_bias_h = True, rnd_seed = None, trainable = True, with_batch=True)

rnn = RNN(name="sgd_rnn", hidden_dim=hidden_dim, input_dim=ip_dim, activation='relu', init_type='one',
          weights_ih=2 * identity_mat, weights_hh=3 * identity_mat, biases_h=np.ones(shape=(hidden_dim,)),
          W_ih_regularizer=None, W_hh_regularizer=None, leak_slope=0.01, clip_threshold=None,
          W_ih_constraint=None, W_hh_constraint=None, b_h_regularizer=None, b_h_constraint=None,
          use_bias_h=True, rnd_seed=None, trainable=True, with_batch=True)

# test_op = rnn.link(ip_tensor, init_hidden=init_state, is_train=False)
test_op = rnn.link(ip_tensor, init_hidden=None, is_train=False)

# get_op = theano.function(inputs=[ip_tensor, init_state],outputs=[test_op])
get_op = theano.function(inputs=[ip_tensor],outputs=[test_op])

rnn2 = RNN(name="sgd_rnn_2", hidden_dim=hidden_dim, input_dim=ip_dim, activation = 'relu', init_type = 'one',
           weights_ih = identity_mat, weights_hh = identity_mat,biases_h = np.ones(shape=(hidden_dim,)),
           W_ih_regularizer = None, W_hh_regularizer = None, leak_slope=0.01, clip_threshold=5.0,
           W_ih_constraint = None, W_hh_constraint = None, b_h_regularizer = None, b_h_constraint = None,
           use_bias_h = True, rnd_seed = None, trainable = True, with_batch=True)

test_op2 = rnn2.link(ip_tensor, init_hidden=test_op[:, -1, :], is_train=False)
# Extracting the last hidden state output for each batch sample

get_op2 = theano.function(inputs=[ip_tensor], outputs=[test_op2])

test_ip = np.ones(shape=(5,ip_dim),dtype=np.float64)

test_ip_batch = [test_ip, 2*test_ip]

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
# Try with different dimensions for hidden, input