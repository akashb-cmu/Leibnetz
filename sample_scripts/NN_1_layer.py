import sys
import theano
import theano.tensor as T
import numpy as np

sys.path.append("/home/akashb/Desktop/Acads/Summer/Leibnetz/Leibnetz")
sys.path.append("/home/akashb/Desktop/Acads/Summer/Leibnetz/Leibnetz/datasets")
from layers import *
from model import *
from optimizers import *
from dataset_utils import *
from regularizers import *
from constraints import *
from model import *


class NN_1_Layer(Model):

    def build_train(self, layer1_ip_dim, layer1_hidden_dim, n_classes, layer1_W_regularizer=None,
                    layer1_W_constraint=None, layer1_b_regularizer=None, layer1_b_constraint=None,
                    layer1_activation='tanh', layer2_W_regularizer=None, layer2_W_constraint=None,
                    layer2_b_regularizer=None, layer2_b_constraint=None, layer2_activation='softmax',
                    layer1_trainable=True, layer2_trainable=True, learning_rate=0.01, rnd_seed=1234):
        """
        This method defines the model architecture for training purposes. It not only defines architecture and persists
        trainable_components, it also configures the model's train_input and train_output tensor variables.
        """
        input_images = T.dmatrix(name='input_img')
        input_labels = T.imatrix(name='input_lbl')

        layer1 = DenseLayer(name="layer1", output_dim=layer1_hidden_dim, input_dim=layer1_ip_dim,
                            activation=layer1_activation, learning_rate=learning_rate, W_regularizer=layer1_W_regularizer,
                            W_constraint=layer1_W_constraint, b_regularizer=layer1_b_regularizer,
                            b_constraint=layer1_b_constraint, trainable=layer1_trainable, rnd_seed=rnd_seed)

        rnd_seed += 9187

        layer2 = DenseLayer(name="layer2", output_dim=n_classes, input_dim=layer1_hidden_dim,
                            activation=layer2_activation, learning_rate=learning_rate, W_regularizer=layer2_W_regularizer,
                            W_constraint=layer2_W_constraint, b_regularizer=layer2_b_regularizer,
                            b_constraint=layer2_b_constraint, trainable=layer2_trainable, rnd_seed=rnd_seed)

        layer1_op = layer1.link(input_images, is_train=True)
        layer2_op = layer2.link(layer1_op, is_train=True)

        self.register_component(layer1)
        self.register_component(layer2)

        self.add_gold_train_output(output_name='layer2_op',gold_train_output_tensor=input_labels)
        self.add_train_input(input_name=input_images.name,train_input_tensor=input_images)
        self.add_train_output(output_name='layer2_op', train_output_tensor=layer2_op)

        """
        Names of model outputs and the corresponding gold output should match exactly
        """

        self.built = True

    def build_test(self):
        """
        This function re-uses the trainable_components persisted in the build_train method (CANNOT CREATE ANY NEW COMPONENTS) and
        defines the model architecture for test time. It also configures the model's test_input and test_output tensor
        variables
        """
        assert self.built, "Please execute build_train first to instantiate components!"
        input_images = self.train_input_tensors['input_img']
        layer1_op = self.all_components['layer1'].link(input_images, is_train=False)
        layer2_op = self.all_components['layer2'].link(layer1_op, is_train=False)
        self.add_test_input(input_name=input_images.name, test_input_tensor=input_images)
        self.add_test_output(output_name="layer2_op", test_output_tensor=layer2_op)
        self.model_test_fn = theano.function(inputs=self.get_ordered_test_input_tensors(),
                                             outputs=self.get_ordered_test_output_tensors())

    def train_model(self, train_input_images, train_output_labels,
                    val_input_images=None, val_output_labels=None,
                    optimizer="batch_gradient_descent", batch_size=1, n_epochs=100, clip_threshold=None,
                    rnd_seed=1234):
        """
        This method accepts actual training matrices from the user and instantiates an optimizer which compiles a train
        function which it uses to train the model
        """
        assert len(self.get_ordered_test_input_tensors()) == 1, "No.of inputs != 1 !"
        assert optimizer == "batch_gradient_descent", "Optimizers other than naive batch gradient descent not" \
                                                      " supported currently!"


        if self.optimizer is None:
            batch_GD_optimizer = BatchGradientDescent(model=self, clip_threshold=clip_threshold)
        else:
            batch_GD_optimizer = self.optimizer
        rnd = np.random.RandomState(rnd_seed)


        batch_GD_optimizer.configure_train_inputs(input_name_to_input_mat_dict=
        {
            self.train_input_tensors['input_img'].name: train_input_images,
        })

        """
        The outputs must be configured before the validation set is configured!
        """

        batch_GD_optimizer.configure_costs_and_gold_outputs(output_name_to_cost_gold_mat_dict=
        {
            'layer2_op': ('categorical_crossentropy',
                          train_output_labels)
        })
        validate = True if val_input_images is not None and val_output_labels is not None \
            else False
        if validate:
            batch_GD_optimizer.configure_val_inputs(val_input_name_to_input_mat_dict=
            {
                self.train_input_tensors['input_img'].name: val_input_images,
            })
            batch_GD_optimizer.configure_val_outputs(val_output_name_to_gold_op_mat_dict=
            {
                'layer2_op': val_output_labels
            })
        # [self.model_train_fn, self.model_val_fn] = batch_GD_optimizer.train(batch_size=batch_size, n_epochs=n_epochs,
        #                                                                     validate=validate,rnd=rnd)

        [self.model_train_fn, self.model_val_fn] = batch_GD_optimizer.train(batch_size=batch_size, n_epochs=n_epochs,
                                                    validate=validate, rnd=rnd, skip_compile=self.optimizer is not None)
        self.optimizer = batch_GD_optimizer

    def apply_model(self, input_images):
        """
        This method accepts test input matrices and returns the outputs
        """
        assert self.model_test_fn is not None, "Please run build_test first!"
        return self.get_ordered_test_output_data(self.model_test_fn(*self.get_ordered_test_input_data({
            self.train_input_tensors['input_img'].name: input_images
        })))



digits=None

(train_images, train_labels) = load_mnist(dataset="training", digits=digits,
                                          path="/home/akashb/Desktop/Acads/Summer/Leibnetz/Leibnetz/datasets/",
                                          asbytes=False, selection=None, return_labels=True, return_indices=False)

(test_images, test_labels) = load_mnist(dataset="testing", digits=digits,
                                        path="/home/akashb/Desktop/Acads/Summer/Leibnetz/Leibnetz/datasets/",
                                        asbytes=False, selection=None, return_labels=True, return_indices=False)

train_images = flatten_mnist_images(train_images)
test_images = flatten_mnist_images(test_images)


input_dim = train_images[0].shape[0]
layer1_dim = 100
layer2_dim = 50
n_epochs = 10
n_classes = 10
batch_size = 1 # None => Gradient descent (whole dataset is a batch)

learning_rate = 0.01

train_labels = get_multiclass_labels(train_labels, n_classes)
test_labels = get_multiclass_labels(test_labels, n_classes)

NN_1 = NN_1_Layer()

NN_1.build_train(layer1_ip_dim=input_dim, layer1_hidden_dim=layer1_dim, n_classes=n_classes, layer1_W_regularizer=None,
                 layer1_W_constraint=None, layer1_b_regularizer=None, layer1_b_constraint=None,
                 layer1_activation='tanh', layer2_W_regularizer=None, layer2_W_constraint=None,
                 layer2_b_regularizer=None, layer2_b_constraint=None, layer2_activation='softmax',
                 layer1_trainable=True, layer2_trainable=True,
                 learning_rate=learning_rate,
                 # learning_rate=learning_rate/(batch_size if batch_size is not None else len(train_images)),
                 rnd_seed=1234)

# NOTE: Optimizer does not average out the gradients when dealing with batches

NN_1.build_test()

NN_1.train_model(train_input_images=train_images, train_output_labels=train_labels,
                 val_input_images=test_images[-1000:], val_output_labels=test_labels[-1000:], batch_size=batch_size,
                 n_epochs=n_epochs, rnd_seed=1234, clip_threshold=None)
# NN_1.train_model(train_input_images=train_images, train_output_labels=train_labels,
#                  val_input_images=None, val_output_labels=None, batch_size=batch_size,
#                  n_epochs=n_epochs, rnd_seed=1234)

model_outputs = NN_1.apply_model(input_images=test_images)

model_test_labels = model_outputs['layer2_op']
# Ideally the model subclass that the user creates should expose a method to get the output of interest

model_test_labels = np.argmax(model_test_labels, axis=1)

model_test_labels = get_multiclass_labels(model_test_labels, n_classes)

test_accuracy = get_multiclass_accuracy(model_test_labels, test_labels)

print("Test accuracy is: %f"%(test_accuracy))

print("Now training on test to demostrate re-usability of copilation graph")

NN_1.train_model(train_input_images=test_images, train_output_labels=test_labels,
                 val_input_images=test_images[-1000:], val_output_labels=test_labels[-1000:], batch_size=batch_size,
                 n_epochs=n_epochs, rnd_seed=1234, clip_threshold=None)

model_outputs = NN_1.apply_model(input_images=test_images)

model_test_labels = model_outputs['layer2_op']
# Ideally the model subclass that the user creates should expose a method to get the output of interest

model_test_labels = np.argmax(model_test_labels, axis=1)

model_test_labels = get_multiclass_labels(model_test_labels, n_classes)

test_accuracy = get_multiclass_accuracy(model_test_labels, test_labels)

print("Test accuracy is: %f"%(test_accuracy))


# VALIDATION DOESN'T WORK!!

# CHECK THAT GRAD CLIP WORKS by setting a very low threshold and confirming that model does badly
