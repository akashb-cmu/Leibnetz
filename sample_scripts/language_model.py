import sys
import theano
import theano.tensor as T
import numpy as np
import cPickle

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

class LangModel(Model):

    def __init__(self, vocab, w_emb_dim, lstm_hidden, model_name="example_lang_model", learning_rate=0.01,
                 init_type='glorot_uniform', use_biases=True,
                 with_batch=False,
                 leak_slope=0.01, clip_threshold=5.0,

                 train_embeddings=True,
                 embedding_constraint=None, embedding_regularizer=None,

                 layer_W_regularizer=None,
                 layer_W_constraint=None, layer_b_regularizer=None, layer_b_constraint=None,
                 layer_activation='softmax',

                 peephole_f=True, peephole_i=True,
                 peephole_o=True, gate_activation='sigmoid',

                 w_hf_regularizer=None, w_xf_regularizer=None, w_cf_regularizer=None, b_f_regularizer=None,
                 w_hu_regularizer=None, w_xu_regularizer=None, b_u_regularizer=None,
                 w_hi_regularizer=None, w_xi_regularizer=None, w_ci_regularizer=None, b_i_regularizer=None,
                 w_ho_regularizer=None, w_xo_regularizer=None, w_co_regularizer=None, b_o_regularizer=None,

                 w_hf_constraint=None, w_xf_constraint=None, w_cf_constraint=None, b_f_constraint=None,
                 w_hu_constraint=None, w_xu_constraint=None, b_u_constraint=None,
                 w_hi_constraint=None, w_xi_constraint=None, w_ci_constraint=None, b_i_constraint=None,
                 w_ho_constraint=None, w_xo_constraint=None, w_co_constraint=None, b_o_constraint=None,
                 lstm_activation='tanh'
                 ):
        super(LangModel, self).__init__()
        assert isinstance(vocab, list), "Vocabulary must be a list of words!"

        # Global model parameters
        self.model_name = model_name
        self.vocab = vocab
        self.w_emb_dim = w_emb_dim
        self.lstm_hidden = lstm_hidden
        self.ip_one_hot = len(vocab)
        self.learning_rate = learning_rate
        self.init_type = init_type
        self.use_biases = use_biases
        self.with_batch = with_batch
        self.leak_slope = leak_slope
        self.clip_threshold = clip_threshold

        # Embedding layer parameters
        self.train_embeddings = train_embeddings
        self.embedding_regularizer = embedding_regularizer
        self.embedding_constraint = embedding_constraint

        # Layer parameters
        self.layer_W_regularizer = layer_W_regularizer
        self.layer_W_constraint = layer_W_constraint
        self.layer_b_regularizer = layer_b_regularizer
        self.layer_b_constraint = layer_b_constraint
        self.layer_activation = layer_activation

        # LSTM specific parameters
        self.peephole_f = peephole_f
        self.peephole_i = peephole_i
        self.peephole_o = peephole_o
        self.gate_activation = gate_activation
        self.lstm_activation = lstm_activation

        self.w_hf_regularizer = w_hf_regularizer
        self.w_xf_regularizer = w_xf_regularizer
        self.w_cf_regularizer = w_cf_regularizer
        self.b_f_regularizer = b_f_regularizer
        self.w_hu_regularizer = w_hu_regularizer
        self.w_xu_regularizer = w_xu_regularizer
        self.b_u_regularizer = b_u_regularizer
        self.w_hi_regularizer = w_hi_regularizer
        self.w_xi_regularizer = w_xi_regularizer
        self.w_ci_regularizer = w_ci_regularizer
        self.b_i_regularizer = b_i_regularizer
        self.w_ho_regularizer = w_ho_regularizer
        self.w_xo_regularizer = w_xo_regularizer
        self.w_co_regularizer = w_co_regularizer
        self.b_o_regularizer = b_o_regularizer

        self.w_hf_constraint = w_hf_constraint
        self.w_xf_constraint = w_xf_constraint
        self.w_cf_constraint = w_cf_constraint
        self.b_f_constraint = b_f_constraint
        self.w_hu_constraint = w_hu_constraint
        self.w_xu_constraint = w_xu_constraint
        self.b_u_constraint = b_u_constraint
        self.w_hi_constraint = w_hi_constraint
        self.w_xi_constraint = w_xi_constraint
        self.w_ci_constraint = w_ci_constraint
        self.b_i_constraint = b_i_constraint
        self.w_ho_constraint = w_ho_constraint
        self.w_xo_constraint = w_xo_constraint
        self.w_co_constraint = w_co_constraint
        self.b_o_constraint = b_o_constraint


        self.optimizer = None

    def build_train(self,

                    embeddings=None,

                    layer_weights=None, layer_biases=None,

                    weights_hf=None, weights_xf=None, weights_cf=None, biases_f=None,
                    weights_hi=None, weights_xi=None, weights_ci=None, biases_i=None,
                    weights_hu=None, weights_xu=None, biases_u=None,
                    weights_ho=None, weights_xo=None, weights_co=None, biases_o=None,

                    rnd_seed=1234
                    ):


        if self.with_batch:
            # self.inputs = T.itensor3("batch_inputs")
            self.inputs = T.imatrix("batch_word_indices")
            self.outputs = T.itensor3("batch_outputs")
        else:
            # self.inputs = T.imatrix("sgd_inputs")
            self.inputs = T.ivector("sgd_word_indices")
            self.outputs = T.imatrix("sgd_outputs")

        self.embed_layer = EmbeddingLayer(layer_name=self.model_name+"_embed_layer",
                                          vocab_size=self.ip_one_hot, embed_size=self.w_emb_dim,
                                          init_type=self.init_type, learning_rate=self.learning_rate,
                                          clip_threshold=self.clip_threshold,
                                          pretrained_embeddings=embeddings,
                                          embedding_constraint=self.embedding_constraint,
                                          embedding_regularizer=self.embedding_regularizer, rnd_seed=rnd_seed,
                                          trainable=self.train_embeddings)
        rnd_seed += 9182

        self.train_ip_projections = self.embed_layer.link(self.inputs, is_train=True)

        self.lstm = LSTM(name=self.model_name+"_batch_lstm", hidden_dim=self.lstm_hidden, input_dim=self.w_emb_dim,
                         peephole_f=self.peephole_f, peephole_i=self.peephole_i,
                         peephole_o=self.peephole_o, gate_activation=self.gate_activation,
                         activation=self.lstm_activation, init_type=self.init_type, with_batch=self.with_batch,
                         leak_slope=self.leak_slope, clip_threshold=self.clip_threshold,
                         weights_hf=weights_hf, weights_xf=weights_xf, weights_cf=weights_cf, biases_f=biases_f,
                         weights_hi=weights_hi, weights_xi=weights_xi, weights_ci=weights_ci, biases_i=biases_i,
                         weights_hu=weights_hu, weights_xu=weights_xu, biases_u=biases_u,
                         weights_ho=weights_ho, weights_xo=weights_xo, weights_co=weights_co, biases_o=biases_o,

                         w_hf_regularizer=self.w_hf_regularizer, w_xf_regularizer=self.w_xf_regularizer,
                         w_cf_regularizer=self.w_cf_regularizer, b_f_regularizer=self.b_f_regularizer,

                         w_hu_regularizer=self.w_hu_regularizer, w_xu_regularizer=self.w_xu_regularizer,
                         b_u_regularizer=self.b_u_regularizer,

                         w_hi_regularizer=self.w_hi_regularizer, w_xi_regularizer=self.w_xi_regularizer,
                         w_ci_regularizer=self.w_ci_regularizer, b_i_regularizer=self.b_i_regularizer,

                         w_ho_regularizer=self.w_ho_regularizer, w_xo_regularizer=self.w_xo_regularizer,
                         w_co_regularizer=self.w_co_regularizer, b_o_regularizer=self.b_o_regularizer,

                         w_hf_constraint=self.w_hf_constraint, w_xf_constraint=self.w_xf_constraint,
                         w_cf_constraint=self.w_cf_constraint,
                         b_f_constraint=self.b_f_constraint,
                         w_hu_constraint=self.w_hu_constraint, w_xu_constraint=self.w_xu_constraint,
                         b_u_constraint=self.b_f_constraint,
                         w_hi_constraint=self.w_hi_constraint, w_xi_constraint=self.w_xi_constraint,
                         w_ci_constraint=self.w_ci_constraint, b_i_constraint=self.b_i_constraint,
                         w_ho_constraint=self.w_ho_constraint, w_xo_constraint=self.w_xo_constraint,
                         w_co_constraint=self.w_co_constraint, b_o_constraint=self.b_o_constraint,

                         use_biases=True, rnd_seed=rnd_seed, trainable=True)

        self.lstm_hids, self.lstm_cs = self.lstm.link(input=self.train_ip_projections, is_train=False)

        rnd_seed+=1349

        self.pred_layer = DenseLayer(name=self.model_name+"_pred_layer", output_dim=self.ip_one_hot,
                                     input_dim=self.lstm_hidden,
                                     activation=self.layer_activation, init_type=self.init_type,
                                     learning_rate=self.learning_rate, leak_slope=self.leak_slope,
                                     clip_threshold=self.clip_threshold, weights=layer_weights, biases=layer_biases,
                                     W_regularizer=self.layer_W_regularizer,
                                     W_constraint=self.layer_W_constraint, b_regularizer=self.layer_b_regularizer,
                                     b_constraint=self.layer_b_constraint, use_bias=self.use_biases, rnd_seed=rnd_seed,
                                     trainable=True)

        self.predictions = self.pred_layer.link(input=self.lstm_hids, is_train=True)

        # Registering components

        self.register_component(self.embed_layer)
        self.register_component(self.lstm)
        self.register_component(self.pred_layer)

        # Configuring train inputs
        self.add_train_input(input_name=self.inputs.name, train_input_tensor=self.inputs)
        # Configuring gold train inputs
        self.add_gold_train_output(output_name=self.pred_layer.layer_name, gold_train_output_tensor=self.outputs)
        # Configuring train model outputs
        self.add_train_output(output_name=self.pred_layer.layer_name, train_output_tensor=self.predictions)

        self.built = True

    def build_test(self):
        """
        Builds the test version of the model. This can differ from the train version of a model if things like dropout
        layers are used. It also compiles the test function.
        """

        assert self.built, "Please build the train version of the model before building the test version!"

        """
        At test time, the input at train time is instead replaced by the output from the previous time step
        """
        if self.with_batch:
            self.test_hid = T.dmatrix("batch_test_hids")
            self.test_c = T.dmatrix("batch_test_hids")
        else:
            self.test_hid = T.dvector("sgd_test_hids")
            self.test_c = T.dvector("sgd_test_cs")

        self.test_ip_projections = self.embed_layer.link(self.inputs, is_train=False)

        self.lstm_test_hids, self.lstm_test_cs = self.lstm.link(input=self.train_ip_projections,
                                                                init_hidden = self.test_hid, init_context = self.test_c,
                                                                is_train=False)

        self.test_predictions = T.argmax(self.pred_layer.link(input=self.lstm_test_hids, is_train=False))


        # Registering test inputs
        self.add_test_input(input_name=self.inputs.name, test_input_tensor=self.inputs)
        self.add_test_input(input_name=self.test_hid.name, test_input_tensor=self.test_hid)
        self.add_test_input(input_name=self.test_c.name, test_input_tensor=self.test_c)

        # Registering test outputs
        self.add_test_output(output_name=self.lstm.layer_name+"_hids", test_output_tensor=self.lstm_test_hids)
        self.add_test_output(output_name=self.lstm.layer_name+"_contexts", test_output_tensor=self.lstm_test_cs)
        self.add_test_output(output_name=self.pred_layer, test_output_tensor=self.test_predictions)

        ordered_test_inputs = self.get_ordered_test_input_tensors()
        ordered_test_outputs = self.get_ordered_test_output_tensors()

        self.model_test_fn = theano.function(inputs=self.get_ordered_test_input_tensors(),
                                             outputs=self.get_ordered_test_output_tensors()
                                             )

        # This test function can now be used to simulate a single step of the decoding process. Since along with the
        # predicted word probabilities, the hidden state and context are also provided, you can seed the states of
        # subsequent states appropriately.

    def train_model(self, train_inputs, train_outputs, val_inputs=None, val_outputs=None,optimizer='batch_gradient_descent', n_epochs=100,
                    clip_threshold=None, rnd_seed=np.random.randint(low=1, high=1000000)):
        batch_size = None
        assert self.built, "Please build the model (at least build the train version) before trying to optimize!"
        assert optimizer=='batch_gradient_descent', "Optimizers other than SGD aren't supported!"

        if clip_threshold is not None:
            assert isinstance(clip_threshold, float) or isinstance(clip_threshold, int) and clip_threshold > 0., \
                "Clip threshold must be a strictly positive value!"

        validate = False
        if all(val is not None for val in [val_inputs, val_outputs]):
            validate = True

        if self.optimizer is None:
            print("Configure the optimizer for the first time")
            batch_optimizer = BatchGradientDescent(model=self, clip_threshold=clip_threshold)
        else:
            batch_optimizer = self.optimizer

        rnd = np.random.RandomState(rnd_seed)
        batch_optimizer.configure_train_inputs(
            input_name_to_input_mat_dict={
                                            self.inputs.name: train_inputs,
                    }
        )

        batch_optimizer.configure_costs_and_gold_outputs(
            output_name_to_cost_gold_mat_dict={
                    self.pred_layer.layer_name  : ("categorical_crossentropy", train_outputs)
                    }
        )

        if validate:
            batch_optimizer.configure_val_inputs(
                val_input_name_to_input_mat_dict={
                    self.inputs.name: train_inputs
                       }
            )

            batch_optimizer.configure_val_outputs(
                val_output_name_to_gold_op_mat_dict={
                    self.pred_layer.layer_name  : train_outputs
                    }
            )

        [self.model_train_fn, self.model_val_fn] = batch_optimizer.train(batch_size=batch_size, n_epochs=n_epochs,
                                                   compile_validation_fn=True, validate=validate,
                                                   rnd=rnd, skip_compile=self.optimizer is not None)

        self.train_fn = batch_optimizer.train_model
        self.optimizer = batch_optimizer

    def validate(self):
        pass

    def apply_model(self):
        pass


# Instantiating the language model

vocab = ["w1","w2","w3"]
w_emb_dim = 50
lstm_hidden = 50
learning_rate = 1.0
n_epochs = 1000
gate_activation = "sigmoid"
init_type='glorot_uniform'


my_lm = LangModel(vocab=vocab, w_emb_dim=w_emb_dim, lstm_hidden=lstm_hidden,
                  model_name="my_first_lm", learning_rate=learning_rate,
                  peephole_f=True, peephole_i=True,
                  peephole_o=True, gate_activation=gate_activation,
                  init_type=init_type, use_biases=True,
                  with_batch=False,
                  leak_slope=0.01, clip_threshold=5.0,

                  train_embeddings=True,
                  embedding_constraint=None, embedding_regularizer=None,

                  layer_W_regularizer=None,
                  layer_W_constraint=None, layer_b_regularizer=None, layer_b_constraint=None,
                  layer_activation='softmax',

                  w_hf_regularizer=None, w_xf_regularizer=None, w_cf_regularizer=None, b_f_regularizer=None,
                  w_hu_regularizer=None, w_xu_regularizer=None, b_u_regularizer=None,
                  w_hi_regularizer=None, w_xi_regularizer=None, w_ci_regularizer=None, b_i_regularizer=None,
                  w_ho_regularizer=None, w_xo_regularizer=None, w_co_regularizer=None, b_o_regularizer=None,

                  w_hf_constraint=None, w_xf_constraint=None, w_cf_constraint=None, b_f_constraint=None,
                  w_hu_constraint=None, w_xu_constraint=None, b_u_constraint=None,
                  w_hi_constraint=None, w_xi_constraint=None, w_ci_constraint=None, b_i_constraint=None,
                  w_ho_constraint=None, w_xo_constraint=None, w_co_constraint=None, b_o_constraint=None,
                  lstm_activation='tanh')

my_lm.build_train(embeddings=None,

                  layer_weights=None, layer_biases=None,

                  weights_hf=None, weights_xf=None, weights_cf=None, biases_f=None,
                  weights_hi=None, weights_xi=None, weights_ci=None, biases_i=None,
                  weights_hu=None, weights_xu=None, biases_u=None,
                  weights_ho=None, weights_xo=None, weights_co=None, biases_o=None,

                  rnd_seed=1234)
my_lm.build_test()

toy_inputs = np.array(
                       [0,1], dtype=np.int32
                     )

toy_outputs = np.array(

                        [[0,1,0],[0,0,1]], dtype=np.int32

                      )

my_lm.train_model(train_inputs=toy_inputs, train_outputs=toy_outputs, val_inputs=toy_inputs, val_outputs=toy_outputs,
                  optimizer='batch_gradient_descent', n_epochs=n_epochs,
                  clip_threshold=None, rnd_seed=np.random.randint(low=1, high=1000000))

print("Done!")