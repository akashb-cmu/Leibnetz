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

digits=[2,9]

(train_images, train_labels) = load_mnist(dataset="training", digits=digits,
                                          path="/home/akashb/Desktop/Acads/Summer/Leibnetz/Leibnetz/datasets/",
                                          asbytes=False, selection=None, return_labels=True, return_indices=False)

(test_images, test_labels) = load_mnist(dataset="testing", digits=digits,
                                          path="/home/akashb/Desktop/Acads/Summer/Leibnetz/Leibnetz/datasets/",
                                          asbytes=False, selection=None, return_labels=True, return_indices=False)

train_labels = binarize_labels(train_labels, digits)
test_labels = binarize_labels(test_labels, digits)

train_images = flatten_mnist_images(train_images)
test_images = flatten_mnist_images(test_images)


layer1_dim = 100
n_epochs = 1
n_classes = 1
batch_size = 1


input_images = T.dmatrix(name='input_img')
input_labels = T.ivector(name='input_lbl')

layer1  = DenseLayer(name="layer1", output_dim=n_classes, input_dim=train_images.shape[1],
                     activation='sigmoid', learning_rate=0.0001)
# TRY APPLYING CONSTRAINTS AND REGULARIZATION!!

model_outputs = layer1.link(input=input_images, is_train=True)

# theano.printing.pydotprint(model_outputs, outfile="model_op.png", var_with_name_simple=True)


bin_log_reg_model = Model()
bin_log_reg_model.register_component(layer1)

SGD_optimizer = BatchGradientDescent(model=bin_log_reg_model, clip=False)
SGD_optimizer.add_inputs({
                            "input_img":(input_images, train_images)
                         })

SGD_optimizer.add_outputs({
                            "model_label":(model_outputs, input_labels, train_labels, "binary_crossentropy")
                          })

SGD_optimizer.train(batch_size=batch_size, n_epochs=n_epochs, validate=False)

outputs_dict = bin_log_reg_model.use_model(inputs_dict={
    "input_img":test_images
})

model_test_labels = np.array(outputs_dict["model_label"] > 0.5, dtype=np.int32)

model_test_labels = model_test_labels.reshape((model_test_labels.shape[0],))

test_accuracy = get_binarized_accuracy(model_test_labels, test_labels)

print("Test accuracy is: %f"%(test_accuracy))

# Try multiclass log reg
# Try multiple objectives - eg. Classification and reconstruction with a shared representation layer in between
# Try using regularization and constraints and confirming that the constraint sactually work!