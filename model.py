import theano
from theano import tensor as T
from theano import config as Tconfig
import abc
import initializers
import activations
import regularizers
import constraints
import losses
import utils
import numpy as np


class Component(object):

    def __init__(self):
        if type(self) is Component:
            raise Exception("Cannot directly instantiate the Component class. Please use a concrete subclass instead.")
        self.component_name = None
        self.learning_rate = None
        self.trainable_params = {}  # Dictionary mapping parameter name to the parameter
        self.trainable_param_names = []  # List of parameter names
        self.regularizers = {}  # Dictionary mapping parmaeter names to regularizers for that parameter
        self.constraints = {}  # Dict that maps tensor: constraints instance.
        self.component_lrates = None

    def get_trainable_params(self):
        # Return the appropriate trainable parameters along with their names as zipped parallel lists
        return (self.trainable_params)

    def get_gradients(self, cost):
        # This method takes a cost theano variable and evaluates the gradients of the cose with respect to all the
        # parameters in this layer. Note that this method does NOT take care of adding the regularization cost of
        # parameters in this layer. Rather it assumes addition of regularization costs have already been added to the
        # input cost variable.
        # This method returns a dictionary mapping param to gradient(param)
        gradients = dict(zip(self.trainable_params.values(), theano.grad(cost=cost,
                                                                       wrt=self.trainable_params.values())))
        return gradients

    def apply_constraints(self, updates_dict, component_name=None):
        # Consumes a dictionary mapping (component, trainable_param_shared_var) to its corresponding update
        # Applies relevant constraints over updates and returns the dictionary mapping param_names to updates

        if component_name is not None:
            assert self.component_name == component_name, "Supplied update dict does not correspond to the component" \
                                                          + self.component_name
        constrained_updates = {}
        for param in updates_dict:
            param_update = updates_dict[param]
            constrained_updates[param] = self.apply_constraint(trainable_param=param, update=param_update)
        return constrained_updates

    def apply_constraint(self, trainable_param, update):
        # Takes a trainable param theano shared variable and its update and returns the update with the relevant
        # constraint applied on it
        assert isinstance(trainable_param, theano.tensor.sharedvar.TensorSharedVariable), "Provided parameter is not a " \
                                                                                          "theano shared variable."
        param_name = trainable_param.name
        assert self.trainable_params.has_key(param_name), "Component " + self.component_name + " does not have " \
                                                          "parameter " + param_name
        return(self.constraints[param_name].constrain(update) if self.constraints.has_key(param_name) else update)

    def get_component_regularization_cost(self, is_train=True):
        # Get regularization cost for this layer only
        component_reg_cost = 0.
        for (param_name, param_value) in self.trainable_params.items():
            component_reg_cost += self.regularizers[param_name].regularize(param=param_value, is_train=is_train)\
                                   if self.regularizers.has_key(param_name)\
                                   else 0.
        return component_reg_cost

    def get_component_name(self):
        assert self.component_name is not None, "Component must have a name"
        return(self.component_name)

    def get_lrates(self):
        # Returns dictionary mapping param to learning_rate(param)
        assert self.learning_rate is not None, "Learning rate not set"
        if self.component_lrates is None:
            param_to_lrates = {}
            for param in self.trainable_params.values():
                param_to_lrates[param] = self.learning_rate
            self.component_lrates = param_to_lrates
        return self.component_lrates



class Model(object):

    """
    This model implementation is relatively anaemic compared to keras since it assumes the developer is responsible for:
    - Generating output tensors through an arbitrarily complex architecture
    - Specifying input tensors to the model
    - Registering components of the architecture with the model
    - Specifying mapping from output tensors to gold tensors
    - Specifying loss between output tensor and gold tensor
    - Specifying optimize

    Model is responsible for:
    - Compiling train/test functions
    - Visualizing the computation graph
    - Saving/Loading the model
    - Training the model
    - Applying model on new data
        - Since exact functionality may vary, any arbitrary model should extend this class and implement this method

    """

    def __init__(self):
        self.components = {}
        self.inputs = {} # Mapping from input_name to input_tensor
        self.outputs = {} # Mapping from output name to output_tensor
        # self.inputs and self.outputs can be used to compile the apply_model function
        # Both self.inputs and self.outputs should be set by an Optimizer object that trains the model
        # This ensures that the apply_model function always generates outputs that the model has been trained to produce
        self.apply_model = None # Compiled theano function to consume inputs and generate outputs
        # Whenever the optimizer re-trains a model, the apply model function is re-compiled to ensure that the model
        # always procudes output that it was most recently trained to produce

    def register_component(self, component):
        assert isinstance(component, Component), "Cannot register a component if it isn't a child of the Component " \
                                                 "class"
        assert self.components.get(component.get_component_name(), None) is None, "Component with same name already " \
                "exists. Possible duplicate name or redundant addition of existing component."
        self.components[component.get_component_name()] = component

    def add_output(self, output_name, output_tensor):
        assert isinstance(output_tensor, theano.tensor.TensorVariable), "Output tensor must be a theano tensor " \
                                                                        "variable (theano.tensor.TensorVariable)"
        self.outputs[output_name] = output_tensor

    def add_input(self, input_name, input_tensor):
        assert isinstance(input_tensor, theano.tensor.TensorVariable), "Input tensor must be a theano tensor " \
                                                                           "variable (theano.tensor.TensorVariable)"
        self.inputs[input_name] = input_tensor

    def get_all_gradients(self, cost):
        component_to_param_to_gradients = {component: {} for component in self.components}
        for component in self.components:
            component_to_param_to_gradients[component].update(self.components[component].get_gradients(cost))
        return component_to_param_to_gradients

    def get_all_regularizer_costs(self, is_train=True):
        reg_cost = 0
        if is_train:
            for component in self.components:
                reg_cost += self.components[component].get_component_regularization_cost(is_train=is_train)
        return reg_cost

    def get_all_lrates(self):
        param_to_lrate = {component:{} for component in self.components}
        for component in self.components:
            param_to_lrate[component].update(self.components[component].get_lrates())
        return(param_to_lrate)

    def apply_all_constraints(self, component_param_updates):
        constrained_component_param_updates = {component:{} for component in component_param_updates}
        for component in component_param_updates:
            constrained_component_param_updates[component].update(self.components[component].apply_constraints(
                updates_dict=component_param_updates[component], component_name=component))
        return constrained_component_param_updates

    def compile_model(self):
        self.apply_model = theano.function(inputs=self.inputs.values(), outputs=self.outputs.values())
        return(self.apply_model)

    def use_model(self, inputs_dict):
        assert self.apply_model is not None, "Model has not been trained yet. Cannot use model unles it has been " \
                                             "trained"
        print("CAUTION: CURRENTLY, the apply_model has compiled the the test function using train computation graph."
              "This is unsuitable is models that use dropout layers for example.")
        inputs = [inputs_dict[input] for input in self.inputs.keys()]
        assert all(isinstance(input, np.ndarray) for input in inputs)
        outputs = self.apply_model(*inputs)
        outputs_dict = {}
        for index, output in enumerate(self.outputs.keys()):
            outputs_dict[output] = outputs[index]
        return(outputs_dict)

# EVENTUALLY IMPLEMENT VISUALIZATION FUNCTION FOR THE MODEL
# ALSO IMPLEMENT MODEL SAVING!!!
# RENAME APPLY MODEL TO TEST_FN