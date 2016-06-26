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

digits=None

(train_images, train_labels) = load_mnist(dataset="training", digits=digits,
                                          path="/home/akashb/Desktop/Acads/Summer/Leibnetz/Leibnetz/datasets/",
                                          asbytes=False, selection=None, return_labels=True, return_indices=False)

(test_images, test_labels) = load_mnist(dataset="testing", digits=digits,
                                        path="/home/akashb/Desktop/Acads/Summer/Leibnetz/Leibnetz/datasets/",
                                        asbytes=False, selection=None, return_labels=True, return_indices=False)

train_images = flatten_mnist_images(train_images)
test_images = flatten_mnist_images(test_images)

layer1_dim = 100
layer2_dim = 50
n_epochs = 1
n_classes = 10
batch_size = 1

train_labels = get_multiclass_labels(train_labels, n_classes)
test_labels = get_multiclass_labels(test_labels, n_classes)


input_images = T.dmatrix(name='input_img')
input_labels = T.imatrix(name='input_lbl')

# layer1  = DenseLayer(name="layer1", output_dim=n_classes, input_dim=train_images.shape[1],
#                      activation='softmax', learning_rate=0.0001)
layer1 = DenseLayer(name="layer1", output_dim=layer1_dim, input_dim=train_images.shape[1],
                    activation='tanh', learning_rate=0.01, W_regularizer=l1l2(l1_reg_coeff=0.01, l2_reg_coeff=0.01), #W_regularizer=eigen_reg(),
                    # W_constraint=None)
                    W_constraint=get_constraint(constraint_type="max_norm", max_norm=0.01))

print("\n\nLayer 1 params:")

param = layer1.W.get_value()
print(param)
# param = param / (epsilon + np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
print(np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))

param = layer1.b.get_value()
print(param)
# param = param / (epsilon + np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
print(np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))

layer2 = DenseLayer(name="layer2", output_dim=n_classes, input_dim=layer1_dim,
                    activation='softmax', learning_rate=0.0001)
# layer2 = DenseLayer(name="layer2", output_dim=layer2_dim, input_dim=layer1_dim,
#                     activation='relu', learning_rate=0.001)

print("\n\nLayer 2 params:")

param = layer2.W.get_value()
print(param)
# param = param / (epsilon + np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
print(np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))

param = layer2.b.get_value()
print(param)
# param = param / (epsilon + np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
print(np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
raw_input("Enter to continue")

# layer3 = DenseLayer(name="layer3", output_dim=n_classes, input_dim=layer2_dim,
#                     activation='softmax', learning_rate=0.0001)
# TRY APPLYING CONSTRAINTS AND REGULARIZATION!!

layer1_outputs = layer1.link(input=input_images, is_train=True)
layer2_outputs = layer2.link(input=layer1_outputs, is_train=True)
# model_outputs = layer3.link(input=layer2_outputs, is_train=True)
model_outputs = layer2_outputs
# theano.printing.pydotprint(model_outputs, outfile="model_op.png", var_with_name_simple=True)


multi_log_reg_model = Model()
multi_log_reg_model.register_component(layer1)
multi_log_reg_model.register_component(layer2)
# multi_log_reg_model.register_component(layer3)

SGD_optimizer = BatchGradientDescent(model=multi_log_reg_model, clip=False)
SGD_optimizer.add_inputs({
    "input_img":(input_images, train_images)
})

SGD_optimizer.add_outputs({
    "model_label":(model_outputs, input_labels, train_labels, "categorical_crossentropy")
})

SGD_optimizer.train(batch_size=batch_size, n_epochs=n_epochs, validate=False)

print("\n\nPost train Layer 1 params:")
param = layer1.W.get_value()
print(param)
# param = param / (epsilon + np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
print(np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))

param = layer1.b.get_value()
print(param)
# param = param / (epsilon + np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
print(np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
raw_input("Enter to continue")

print("\n\nPost train Layer 2 params:")

param = layer2.W.get_value()
print(param)
# param = param / (epsilon + np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
print(np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))

param = layer1.b.get_value()
print(param)
# param = param / (epsilon + np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
print(np.sqrt(np.sum(np.square(param), axis=0, keepdims=True)))
raw_input("Enter to continue")

outputs_dict = multi_log_reg_model.use_model(inputs_dict={
    "input_img":test_images
})

model_test_labels = np.argmax(outputs_dict["model_label"], axis=1)
model_test_labels = get_multiclass_labels(model_test_labels, n_classes)

test_accuracy = get_multiclass_accuracy(model_test_labels, test_labels)

print("Test accuracy is: %f"%(test_accuracy))

# Try multiclass log reg
# Try multiple objectives - eg. Classification and reconstruction with a shared representation layer in between
# Try using regularization and constraints and confirming that the constraint sactually work!
# Try sparse softmax to confirm that it works!
# Try different activations
# Try saving and loading model, initializing params etc.