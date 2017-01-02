import theano.tensor as T
from layers import *
from utils import eprint


class Component(object):
    """
    Parent class of all layers/neural network constituents that can be used in a Model. This class is essentially a
    contract that specifies the expectations in terms of functionalities from a network constituent.

    NOTE: THIS CLASS SHOULD PROBABLY BE AN ABSTRACT CLASS!
    """

    __metaclass__ = abc.ABCMeta

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

        self.trainable = True

    def get_trainable_params(self):
        # Return the appropriate trainable parameters along with their names as zipped parallel lists
        return (self.trainable_params)

    def get_gradients(self, cost, clip_threshold=None):
        # This method takes a cost theano variable and evaluates the gradients of the cose with respect to all the
        # parameters in this layer. Note that this method does NOT take care of adding the regularization cost of
        # parameters in this layer. Rather it assumes addition of regularization costs have already been added to the
        # input cost variable.
        # This method returns a dictionary mapping param to gradient(param)
        # gradients = dict(zip(self.trainable_params.values(), theano.grad(cost=cost,
        #                                                                wrt=self.trainable_params.values())))
        if clip_threshold is None:
            gradients = dict( zip( self.trainable_params.values(),
                                   [theano.grad(cost=cost, wrt=param) for param in self.trainable_params.values()] ) )
        else: # clip the gradients before updates are applied!
            assert clip_threshold > 0., "Clip threshold MUST be a positive value for component " + self.component_name
            gradients = dict(zip(self.trainable_params.values(),
                                 [theano.grad(cost=cost, wrt=param).clip(a_min=-clip_threshold,a_max=clip_threshold)
                                                         for param in self.trainable_params.values()]))
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
    - Registering trainable_components of the architecture with the model
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
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.trainable_components = {}
        self.all_components = {}
        self.train_input_tensors = {} # Mapping from input_name to input_tensor
        self.train_output_tensors = {} # Mapping from output name to output_tensor
        self.test_input_tensors = {}  # Mapping from input_name to input_tensor
        self.test_output_tensors = {}  # Mapping from output name to output_tensor
        self.gold_train_outputs_tensors = {} # Mapping from output name to the gold train tensor wrt which loss must be
        # evaluated. Loss between the train model output and the train gold output is not stored by the model since
        # that is relevant only to the optimizer.
        self.train_outputs_ordering = None
        self.train_inputs_ordering = None
        self.test_outputs_ordering = None
        self.test_inputs_ordering = None

        self.model_test_fn = None # Compiled theano function to consume test_input_tensors and generate test_output_tensors
        self.built = False
        self.optimizer = None

    def register_component(self, component):
        assert isinstance(component, Component), "Cannot register a component if it isn't a child of the Component " \
                                                 "class"
        assert self.trainable_components.get(component.get_component_name(), None) is None, "Trainable component with "\
               "name " + str(component.component_name)  + "already exists. Possible duplicate name or redundant " \
               "addition of existing component."
        assert self.all_components.get(component.component_name, None) is None, "Non trainable Component " \
               "with name " + str(component.component_name) + "already exists. Possible duplicate name or redundant " \
               "addition of existing component."
        if component.trainable:
            print("Registered trainable component %s"%(component.component_name))
            self.trainable_components[component.get_component_name()] = component
        else:
            print("Registered non-trainable component %s"%(component.component_name))
        self.all_components[component.component_name] = component

    def add_train_output(self, output_name, train_output_tensor):
        assert isinstance(train_output_tensor, theano.tensor.TensorVariable), "Output tensor must be a theano tensor " \
                                                                        "variable (theano.tensor.TensorVariable)"
        self.train_output_tensors[output_name] = train_output_tensor

    def add_gold_train_output(self, output_name, gold_train_output_tensor):
        assert isinstance(gold_train_output_tensor, theano.tensor.TensorVariable), "Output tensor must be a theano " \
               "tensor variable (theano.tensor.TensorVariable)!"
        self.gold_train_outputs_tensors[output_name] = gold_train_output_tensor

    def add_train_input(self, input_name, train_input_tensor):
        assert isinstance(train_input_tensor, theano.tensor.TensorVariable), "Input tensor must be a theano tensor " \
                                                                           "variable (theano.tensor.TensorVariable)"
        self.train_input_tensors[input_name] = train_input_tensor

    def add_test_output(self, output_name, test_output_tensor):
        assert isinstance(test_output_tensor, theano.tensor.TensorVariable), "Output tensor must be a theano tensor " \
                                                                        "variable (theano.tensor.TensorVariable)"
        self.test_output_tensors[output_name] = test_output_tensor

    def add_test_input(self, input_name, test_input_tensor):
        assert isinstance(test_input_tensor, theano.tensor.TensorVariable), "Input tensor must be a theano tensor " \
                                                                       "variable (theano.tensor.TensorVariable)"
        self.test_input_tensors[input_name] = test_input_tensor


    def get_all_gradients(self, cost, clip_threshold=None):
        component_to_param_to_gradients = {component: {} for component in self.trainable_components}
        for component in self.trainable_components:
            component_to_param_to_gradients[component].update(self.trainable_components[component].get_gradients(cost,
                                                                                         clip_threshold=clip_threshold))
        return component_to_param_to_gradients

    def get_all_regularizer_costs(self, is_train=True):
        # reg_cost = 0
        eprint("Currently regularizer costs are reported at a per component level rather than a per parameter level!")
        reg_cost = []
        reg_cost_names = []
        if is_train:
            for component in self.trainable_components:
                comp_reg_cost = self.trainable_components[component].get_component_regularization_cost(is_train=is_train)
                if comp_reg_cost != 0.:
                    reg_cost.append(self.trainable_components[component].get_component_regularization_cost(is_train=is_train))
                    reg_cost_names.append(self.trainable_components[component].component_name + "_reg_cost")
        return reg_cost, reg_cost_names

    def get_all_lrates(self):
        param_to_lrate = {component:{} for component in self.trainable_components}
        for component in self.trainable_components:
            param_to_lrate[component].update(self.trainable_components[component].get_lrates())
        return(param_to_lrate)

    def apply_all_constraints(self, component_param_updates):
        constrained_component_param_updates = {component:{} for component in component_param_updates}
        for component in component_param_updates:
            constrained_component_param_updates[component].update(self.trainable_components[component].apply_constraints(
                updates_dict=component_param_updates[component], component_name=component))
        return constrained_component_param_updates

    def compile_model(self):
        self.sanity_check_test_outputs_ordering()
        self.sanity_check_test_inputs_ordering()
        # self.model_test_fn = theano.function(inputs=self.test_input_tensors.values(),
        #                                      outputs=self.test_output_tensors.values())
        self.model_test_fn = theano.function(inputs=self.get_ordered_test_input_tensors(),
                             outputs=self.get_ordered_test_output_tensors())
        return(self.model_test_fn)

    @abc.abstractmethod
    def build_train(self):
        """
        This method defines the model architecture for training purposes. It not only defines architecture and persists
        trainable_components, it also configures the model's train_input and train_output tensor variables.
        """
        assert False, "Implement this method!"

    @abc.abstractmethod
    def build_test(self):
        """
        This function re-uses the trainable_components persisted in the build_train method (CANNOT CREATE ANY NEW COMPONENTS) and
        defines the model architecture for test time. It also configures the model's test_input and test_output tensor
        variables
        """
        assert False, "Implement this method!"

    @abc.abstractmethod
    def train_model(self):
        """
        This method accepts actual training matrices from the user and instantiates an optimizer which compiles a train
        function which it uses to train the model
        """
        assert False, "Implement this method!"

    @abc.abstractmethod
    def apply_model(self):
        """
        This method accepts test input matrices and returns the outputs
        """
        assert False, "Implement thsi method!"

    def sanity_check_test_outputs_ordering(self):
        test_output_tensor_set = set(self.test_output_tensors.keys())
        assert len(test_output_tensor_set) > 0, "No test_outputs configured in " \
                                                                       "model!"

        assert test_output_tensor_set.intersection( set(
            self.train_output_tensors.keys()) ) == test_output_tensor_set, \
            "Model train_output_tensors and model test_output_tensors are out of sync"
        assert test_output_tensor_set.intersection(set(
            self.gold_train_outputs_tensors.keys()) ) == test_output_tensor_set, \
            "Model test_output_tensors and model gold_output_tensors are out of sync! This also means the model" \
            " model train_output_tensors and model gold_output_tensors are out of sync!"
        if self.test_outputs_ordering is None:
            self.test_outputs_ordering = sorted(self.test_output_tensors.keys())
        else:
            ordering_set = set(self.test_outputs_ordering)
            assert all(output in ordering_set for output in
                       self.test_output_tensors.keys()), "test output ordering does not include all model " \
                                                                "outputs!"

    def sanity_check_test_inputs_ordering(self):
        test_input_tensor_set = set(self.test_input_tensors.keys())
        assert len(test_input_tensor_set) > 0, "No test_inputs configured in " \
                                                "model!"

        assert test_input_tensor_set.intersection(set(
            self.train_input_tensors.keys())) == test_input_tensor_set, \
            "Model train_input_tensors and model test_input_tensors are out of sync"
        if self.test_inputs_ordering is None:
            self.test_inputs_ordering = sorted(self.test_input_tensors.keys())
        else:
            ordering_set = set(self.test_inputs_ordering)
            assert all(input in ordering_set for input in
                       self.test_input_tensors.keys()), "test input ordering does not include all model " \
                                                         "inputs!"

    def get_ordered_test_input_tensors(self):
        self.sanity_check_test_inputs_ordering()
        return [self.test_input_tensors[key] for key in self.test_inputs_ordering]

    def get_ordered_test_output_tensors(self):
        self.sanity_check_test_outputs_ordering()
        return [self.test_output_tensors[key] for key in self.test_outputs_ordering]

    def get_ordered_test_input_data(self, input_data_dict):
        self.sanity_check_test_inputs_ordering()
        assert set(input_data_dict.keys()).intersection(set(self.test_inputs_ordering)) == set(self.test_input_tensors),\
            "Input data dict does not have necessary args!"
        return [input_data_dict[key] for key in self.test_inputs_ordering]

    def get_ordered_test_output_data(self, test_fn_output):
        self.sanity_check_test_outputs_ordering()
        assert len(test_fn_output) == len(self.test_outputs_ordering), \
            "Too few outputs provided!"
        return {self.test_outputs_ordering[i]: test_fn_output[i] for i in range(len(self.test_outputs_ordering))}


# # EVENTUALLY IMPLEMENT VISUALIZATION FUNCTION FOR THE MODEL
# # ALSO IMPLEMENT MODEL SAVING!!!
# # RENAME APPLY MODEL TO TEST_FN