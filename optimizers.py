import theano
import theano.tensor as T
import numpy as np
from model import Model
import losses
import abc
from collections import OrderedDict

class Optimizer(object):
    __metaclass__=abc.ABCMeta
    def __init__(self, model, clip=False, clip_threshold=5.):
        assert isinstance(model, Model), "Model input to optimizer must be an instance of the Model class!"
        assert clip_threshold > 0., "Clip threshold must be positive!"
        self.model = model
        self.clip = clip # This has to be set as a variable of the optimizer class since updates will change based on
        # the clip setting and caching updates as another variable of the class will be pointless
        self.clip_threshold = clip_threshold
        self.updates = None
        self.input_name_to_tensor_gold_mat_dict = {} # Maps name of an input to its gold tensor representation and the
        # actual input to be used for training
        self.val_input_name_to_gold_mat_dict = {} # Maps name of val input to its gold tensor representation and the
        # actual val input to be used for validation
        self.val_output_name_to_gold_mat_dict = {}
        self.output_name_to_model_op_gold_op_gold_mat_cost_dict = {} # Maps name of an output to its gold tensor representation, the
        # actual output to be used for training and the cost function to be applied
        self.validate_model = None
        self.train_model = None
    """
    Optimizer should support clip (T/F), clip_val, clip_norm, and any other optimizer specific options as arguments to
    the optimize function
    """

    @abc.abstractmethod
    def get_updates(self):
        """
        Applies optimizer specific logic to derive updates using gradients for all model components
        """
        raise NotImplementedError("get_updates has not been implemented")

    @abc.abstractmethod
    def compile_model(self):
        """
        Compiles train and validate functions
        """
        raise NotImplementedError("train method has not been implemented")

    @abc.abstractmethod
    def train(self):
        """
        Consumes inputs, batches it and then applies the train function
        """
        raise NotImplementedError("train method has not been implemented")

    @abc.abstractmethod
    def validate(self):
        """
        Consumes validation input and applies the validation function. Note that the validation set is not batched or
        shuffled.
        """
        raise NotImplementedError("validate method has not been implemented")

    def get_model_output_tensors(self):
        assert len(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.values()) > 0, "No outputs configured."
        assert set(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.keys()) == set(
            self.model.outputs.keys()), \
            "Optimizer outputs and model outputs are out of sync"
        return self.model.outputs.values()  # Has same order as self.model.outputs.keys()

    def get_gold_output_tensors(self):
        assert len(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.values()) > 0, "No outputs configured."
        assert set(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.keys()) == set(
            self.model.outputs.keys()), \
            "Optimizer outputs and model outputs are out of sync"
        return [self.output_name_to_model_op_gold_op_gold_mat_cost_dict[output][1] for output in
                self.model.outputs.keys()]

    def get_train_outputs(self):
        assert len(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.values()) > 0, "No outputs configured."
        assert set(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.keys()) == set(
            self.model.outputs.keys()), \
            "Optimizer outputs and model outputs are out of sync"
        return [self.output_name_to_model_op_gold_op_gold_mat_cost_dict[output][2] for output in
                self.model.outputs.keys()]

    def get_output_costs(self):
        assert len(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.values()) > 0, "No outputs configured."
        assert set(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.keys()) == set(
            self.model.outputs.keys()), \
            "Optimizer outputs and model outputs are out of sync"
        return [self.output_name_to_model_op_gold_op_gold_mat_cost_dict[output][3] for output in
                self.model.outputs.keys()]

    def get_model_input_tensors(self):
        assert len(self.input_name_to_tensor_gold_mat_dict.values()) > 0, "No inputs configured."
        assert set(self.input_name_to_tensor_gold_mat_dict.keys()) == set(self.model.inputs.keys()), "Optimizer " \
                                                                                                     "inputs " \
                                                                                                     "and model " \
                                                                                                     "inputs are out " \
                                                                                                     "of sync"
        return self.model.inputs.values()  # Has same ordering as self.model.inputs.keys()

    def get_train_inputs(self):
        assert len(self.input_name_to_tensor_gold_mat_dict.values()) > 0, "No inputs configured."
        assert set(self.input_name_to_tensor_gold_mat_dict.keys()) == set(self.model.inputs.keys()), "Optimizer " \
                                                                                                     "inputs " \
                                                                                                     "and model " \
                                                                                                     "inputs are out " \
                                                                                                     "of sync"
        return [self.input_name_to_tensor_gold_mat_dict[input][1] for input in self.model.inputs.keys()]

    def get_val_inputs(self):
        assert len(self.val_input_name_to_gold_mat_dict.values()) > 0, "No val inputs configured."
        assert set(self.val_input_name_to_gold_mat_dict.keys()) == set(
            [self.input_name_to_tensor_gold_mat_dict.keys()]), \
            "Validation inputs don't match the train inputs"
        assert set(self.val_input_name_to_gold_mat_dict.keys()) == set([self.model.inputs.keys()]), "Validation " \
                                                                                                    "inputs don't " \
                                                                                                    "match the model " \
                                                                                                    "inputs"
        return [self.val_input_name_to_gold_mat_dict[input] for input in self.model.inputs.keys()]

    def get_val_outputs(self):
        assert len(self.val_output_name_to_gold_mat_dict.values()) > 0, "No val outputs configured."
        assert set(self.val_output_name_to_gold_mat_dict.keys()) == set(
            [self.output_name_to_model_op_gold_op_gold_mat_cost_dict.keys()]), \
            "Validation outputs don't match the train outputs"
        assert set(self.val_output_name_to_gold_mat_dict.keys()) == set([self.model.outputs.keys()]), "Validation " \
                                                                                                      "outputs don't " \
                                                                                                      "match the " \
                                                                                                      "model " \
                                                                                                      "outputs"
        return [self.val_output_name_to_gold_mat_dict[output] for output in self.model.outputs.keys()]



    def add_output(self, output_name, model_output_tensor, gold_output_tensor, gold_output_mat, loss_type):
        assert isinstance(model_output_tensor, theano.tensor.TensorVariable), "Model output tensor is not a " \
                                                                                  "theano.tensor.TensorVariable"
        assert isinstance(gold_output_tensor, theano.tensor.TensorVariable), "Gold output tensor is not a " \
                                                                                 "theano.tensor.TensorVariable"
        assert isinstance(gold_output_mat, np.ndarray), "Provided gold outputs is not a numpy.ndarray"
        if len(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.values()) > 0:
            assert gold_output_mat.shape[0] == \
                   self.output_name_to_model_op_gold_op_gold_mat_cost_dict.values()[0][2].shape[0], "New output has " \
                                                                                                    "different number " \
                                                                                                    "of samples."
        self.output_name_to_model_op_gold_op_gold_mat_cost_dict[output_name] = (model_output_tensor, gold_output_tensor,
                                                                                gold_output_mat,
                                                                                losses.get_loss(loss_type=loss_type,
                                                                                                y_pred=model_output_tensor,
                                                                                                y_actual=gold_output_tensor))
        self.model.add_output(output_name=output_name, output_tensor=model_output_tensor)

    def add_outputs(self, output_name_to_model_op_gold_op_gold_op_mat_cost_dict):
        """
        :param output_name_to_model_op_gold_op_gold_op_mat_cost_dict: Dictionary mapping an output name to the model
        output tensor,
                                                          gold output tensor, actual gold training matrix and loss_type
                                                          the output
        """
        for output_name in output_name_to_model_op_gold_op_gold_op_mat_cost_dict:
            model_op_tensor, gold_op_tensor, gold_op_mat, loss_type = \
            output_name_to_model_op_gold_op_gold_op_mat_cost_dict[
                output_name]
            self.add_output(output_name=output_name, model_output_tensor=model_op_tensor,
                            gold_output_tensor=gold_op_tensor, gold_output_mat=gold_op_mat, loss_type=loss_type)

    def add_input(self, input_name, input_tensor, actual_input):
        assert isinstance(input_tensor, theano.tensor.TensorVariable), "Input tensor is not of type" \
                                                                           " theano.tensor.TensorVariable"
        assert isinstance(actual_input, np.ndarray), "Provided actual input is not np.ndarray"
        if len(self.input_name_to_tensor_gold_mat_dict.values()) > 0:
            assert actual_input.shape[0] == self.input_name_to_tensor_gold_mat_dict.values()[0][1].shape[0], \
                "New input" \
                                                                                                             "has " \
                "different number of samples."
        assert isinstance(actual_input, np.ndarray), "Supplied train input is not a numpy.ndarray!"
        self.input_name_to_tensor_gold_mat_dict[input_name] = (input_tensor, actual_input)
        self.model.add_input(input_name=input_name, input_tensor=input_tensor)

    def add_inputs(self, input_tensor_to_input_dict):
        """
        :param input_tensor_to_input_dict: Dictionary mapping input name to the input_tensor and the actual training
                                           input
        """
        for input in input_tensor_to_input_dict:
            input_tensor, actual_input = input_tensor_to_input_dict[input]
            self.add_input(input_name=input, input_tensor=input_tensor, actual_input=actual_input)

    def add_val_input(self, input_name, actual_val_input):
        assert self.input_name_to_tensor_gold_mat_dict.has_key(input_name), "Validation input " + input_name + " is " \
                                                                                                               "not" \
                                                                                           "a valid input to the model."
        assert isinstance(actual_val_input, np.ndarray), "Supplied validation input is not a numpy.ndarray!"
        assert actual_val_input.shape[-1] == self.input_name_to_tensor_gold_mat_dict[input_name][1].shape[-1],"Feature"\
                " dimensions of provided validation input " + input_name + " don't match the corresponding train input"\
                                                                           " dimension"
        if len(self.val_input_name_to_gold_mat_dict.values()) > 0:
            assert actual_val_input.shape[0] == self.val_input_name_to_gold_mat_dict.values()[0].shape[0], \
                "New val input has different number of samples."
        self.val_input_name_to_gold_mat_dict[input_name] = actual_val_input

    def add_val_inputs(self, val_input_tensor_to_input_dict):
        """
        :param val_input_tensor_to_input_dict: Dictionary mapping input name to the actual training
                                           input
        """
        for input in val_input_tensor_to_input_dict:
            actual_input = val_input_tensor_to_input_dict[input]
            self.add_val_input(input_name=input, actual_val_input=actual_input)

    def add_val_output(self, output_name, gold_val_output_mat):
        assert self.output_name_to_model_op_gold_op_gold_mat_cost_dict.has_key(output_name), "Validation output " \
                                                                                             + output_name + " is not " \
                                                                                                             "a valid " \
                                                                                                             "output " \
                                                                                                             "to the " \
                                                                                                             "model."
        assert isinstance(gold_val_output_mat, np.ndarray), "Supplied validation output data is not of type np.ndarray"
        assert gold_val_output_mat.shape[-1] == \
               self.output_name_to_model_op_gold_op_gold_mat_cost_dict[output_name][2].shape[-1], \
            "Output dimension of provided validation output " + output_name + " don't match the corresponding train" \
                                                                              " output dimension"
        if len(self.val_output_name_to_gold_mat_dict.values()) > 0:
            assert gold_val_output_mat.shape[0] == \
                   self.val_output_name_to_gold_mat_dict.values()[0].shape[0], "New output has different number of " \
                                                                               "samples."

        self.val_output_name_to_gold_mat_dict[output_name] = gold_val_output_mat

    def add_val_outputs(self, val_output_name_to_gold_op_mat_dict):
        """
        :param val_output_name_to_gold_op_mat_dict: Dictionary mapping an output name to the model output tensor,
                                                          gold output tensor, actual gold training matrix and loss_type
                                                          the output
        """
        for output_name in val_output_name_to_gold_op_mat_dict:
            self.add_val_output(output_name=output_name,
                                gold_val_output_mat=val_output_name_to_gold_op_mat_dict[output_name])

    def get_cost(self, is_train=True):
        cost = T.sum(self.get_output_costs())
        reg_cost = self.model.get_all_regularizer_costs(is_train=is_train)
        return cost + reg_cost



class BatchGradientDescent(Optimizer):

    def __init__(self, model, clip=False, clip_threshold=5.):
        super(BatchGradientDescent, self).__init__(model=model, clip=clip, clip_threshold=clip_threshold)

    def get_updates(self):
        if self.updates is not None:
            return self.updates
        # train_cost = self.model.get_cost(is_train=True)
        train_cost = self.get_cost(is_train=True)
        if self.clip:
            train_cost = theano.gradient.grad_clip(x=train_cost, lower_bound=-1. * self.clip_threshold,
                                                   upper_bound=1. * self.clip_threshold)
        component_to_param_to_gradients = self.model.get_all_gradients(cost=train_cost)
        """
        Dictionary mapping
        {
            component: {
                            param: gradient
                       }
        }
        """
        component_param_to_lrate = self.model.get_all_lrates()
        """
            Dictionary mapping
            {
                component: {
                                param: lrate
                           }
            }
        """
        # dictionary mapping (component_name, param) to learning_rate(param)
        component_param_updates = {component:{} for component in component_to_param_to_gradients}
        """
            Dictionary mapping
            {
                component: {
                                param: update
                           }
            }
        """
        for component in component_to_param_to_gradients:
            for param in component_to_param_to_gradients[component]:
                component_param_updates[component][param] = param - (component_param_to_lrate[component][param] *
                                                                     component_to_param_to_gradients[component][param])

        return(self.model.apply_all_constraints(component_param_updates=component_param_updates))

    def compile_model(self, validate=True):
        gd_updates = self.get_updates()
        # updates_dict = {}
        updates_dict = OrderedDict()
        for component in gd_updates:
            updates_dict.update(gd_updates[component])
        train_cost = self.get_cost(is_train=True)
        self.train_model = theano.function(inputs=self.get_model_input_tensors()+self.get_gold_output_tensors(),
                                           outputs=self.get_model_output_tensors() + [train_cost],
                                           updates=updates_dict)
        val_cost = self.get_cost(is_train=False)
        if validate:
            print("CAUTION: Currently optimizer compiles validation function using computation graph for train. This "
                  "is not suitable for models that use dropout layers for example")
            self.validate_model = theano.function(inputs=self.get_model_input_tensors()+self.get_gold_output_tensors(),
                                                outputs=self.get_model_output_tensors() + [val_cost])
        self.model.compile_model()

    def train(self, batch_size=None, n_epochs=100, validate=True):
        self.compile_model(validate=validate)
        train_inputs = self.get_train_inputs()
        train_outputs = self.get_train_outputs()
        input_size = train_inputs[0].shape[0]
        output_size = train_outputs[0].shape[0]
        assert all(input.shape[0] == input_size for input in train_inputs), "Mismatch in number of input samples of" \
                                                                            " configured actual inputs"
        assert all(output.shape[0] == output_size for output in train_outputs), "Mismatch in number of input samples of" \
                                                                                " configured actual outputs"
        assert output_size == input_size, "Number of output samples and number of input samples don't match"
        if validate:
            validation_inputs = self.get_val_inputs()
            validation_outputs = self.get_val_outputs()
            val_input_size = validation_inputs[0].shape[0]
            val_output_size = validation_outputs[0].shape[0]
            assert all(val_input.shape[0] == val_input_size for val_input in validation_inputs),\
                "Mismatch in number of input samples of configured validation inputs"
            assert all(val_output.shape[0] == val_output_size for val_output in validation_inputs), \
                "Mismatch in number of output samples of configured validation outputs"
            assert val_input_size == val_output_size, "Validation input and validation output sizes don't match"
        if batch_size is not None:
            assert isinstance(batch_size, int), "Batch size is not an integer"
        assert isinstance(n_epochs, int), "Number of epochs must be an integer"
        train_index_shuffle_vect = np.array([i for i in range(input_size)])
        for epoch in range(n_epochs):
            print("Epoch %d"%(epoch + 1))
            if batch_size is not None:
                np.random.shuffle(train_index_shuffle_vect)
                tot_epoch_cost = 0.
                tot_epoch_val_cost = 0.
                tot_batches = 0.
                for batch_start_index in range(0, input_size, batch_size):
                    tot_batches += 1.
                    batch_indices = train_index_shuffle_vect[batch_start_index: batch_start_index + batch_size]
                    train_batch_inputs = [input[[batch_indices]] for input in train_inputs]
                    train_batch_outputs = [output[[batch_indices]] for output in train_outputs]
                    train_fn_outputs = self.train_model(*(train_batch_inputs + train_batch_outputs))
                    tot_epoch_cost += train_fn_outputs[-1]
                    # print("Training loss is : %f"%(train_fn_outputs[-1]))
                    if validate:
                        val_fn_outputs = self.validate_model(*(validation_inputs+validation_outputs))
                        tot_epoch_val_cost += val_fn_outputs[-1]
                        # print("Validation loss is : %f"%(val_fn_outputs[-1]))
                print("Avg. epoch cost is : %f"%(tot_epoch_cost/tot_batches))
                if validate:
                    print("Avg. epoch val cost is : %f"%(tot_epoch_val_cost/tot_batches))
            else:
                train_fn_outputs = self.train_model(*(train_inputs + train_outputs))
                print("Training loss is : %f"%(train_fn_outputs[-1]))
                if validate:
                    val_fn_outputs = self.validate_model(*(validation_inputs+validation_outputs))
                    print("Validation loss is : %f"%(val_fn_outputs[-1]))
        print("Done training!")

    def validate(self, validation_inputs, validation_outputs):
        # validation_inputs = self.get_val_inputs()
        # validation_outputs = self.get_val_outputs()
        assert len(validation_inputs) == len(self.input_name_to_tensor_gold_mat_dict.keys()), \
            "Inappropriate number of inputs provided"
        assert len(validation_outputs) == len(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.keys()), \
            "Inapprpriate number of outputs provided"
        val_input_size = validation_inputs[0].shape[0]
        val_output_size = validation_outputs[0].shape[0]
        assert all(val_input.shape[0] == val_input_size for val_input in validation_inputs), \
            "Mismatch in number of input samples of configured validation inputs"
        assert all(val_output.shape[0] == val_output_size for val_output in validation_inputs), \
            "Mismatch in number of output samples of configured validation outputs"
        assert val_input_size == val_output_size, "Validation input and validation output sample sizes don't match"
        assert self.validate_model is not None, "Validate function has not been compiled yet!"
        val_fn_outputs = self.validate_model(*(validation_inputs + validation_outputs))
        print("Validation loss is : %f"%(val_fn_outputs[-1]))


    # MIGRATE FUNCTIONS FROM BatchGradientDescent TO Optimizer parent class as appropriate
    # COMPLETE THE VALIDATE FUNCTION!! ---> chekcing types of arguments, sufficiency of inputs etc.
    # Implement methods to clear out all data structures
    # Ensure order of outputs and inputs is constant
    # CONFIRM ALL OF THE BELOW HAVE BEEN SUITABLY CHANGED TO WORK WITH THE OPTIMIZER CLASS