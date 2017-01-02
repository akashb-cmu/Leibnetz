from model import Model
import losses
from collections import OrderedDict
from layers import *
from utils import  eprint

class Optimizer(object):
    __metaclass__=abc.ABCMeta
    def __init__(self, model, clip_threshold=None):
        assert isinstance(model, Model), "Model input to optimizer must be an instance of the Model class!"
        self.model = model
        self.clip = True if clip_threshold is not None else False # This has to be set as a variable of the optimizer
        # class since updates will change based on
        # the clip setting and caching updates as another variable of the class will be pointless
        if self.clip:
            assert clip_threshold > 0., "Clip threshold must be positive!"
        self.clip_threshold = clip_threshold
        self.updates = None
        self.input_name_to_gold_mat_dict = {} # Maps name of an input to the actual corresponding input to be used for
        # training. The corresponding train tensors are obtained from the model itself.
        self.val_input_name_to_gold_mat_dict = {} # Maps name of val input to the  actual val input to be used for
        # validation. The corresponding train tensors are the same as the input tensors
        self.val_output_name_to_gold_mat_dict = {}# Maps name of val output to the actual val input to be used for
        # validation. The corresponding train tensors are the same as the input tensors
        self.output_name_to_cost_gold_mat_dict = {} # Maps name of an output to the loss to apply between the model's
        # train_output tensor and its train_gold_output tensor. The train_output and train_gold_output tensors are
        # obtained from the model itself
        self.validate_model = None
        self.train_model = None
        self.train_cost_names = None
        self.val_cost_names = None
    """
    Optimizer should support clip (T/F), clip_val, clip_norm, and any other optimizer specific options as arguments to
    the optimize function
    """

    @abc.abstractmethod
    def get_updates(self):
        """
        Applies optimizer specific logic to derive updates using gradients for all model trainable_components
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
        Consumes train_input_tensors, batches it and then applies the train function
        """
        raise NotImplementedError("train method has not been implemented")

    @abc.abstractmethod
    def validate(self):
        """
        Consumes validation input and applies the validation function. Note that the validation set is not batched or
        shuffled.
        """
        raise NotImplementedError("validate method has not been implemented")

    ####################################################################################################################
    ######################################## S A N I T Y   C H E C K S #################################################
    ####################################################################################################################

    def sanity_check_train_outputs_ordering(self, validate=False):
        assert len(self.output_name_to_cost_gold_mat_dict.keys()) > 0, "No train_outputs configured in " \
                                                                         "optimizer!"

        assert set(self.output_name_to_cost_gold_mat_dict.keys()) == set(
            self.model.train_output_tensors.keys()), \
            "Optimizer train_output_tensors and model train_output_tensors are out of sync"
        if validate:
            assert set(self.val_output_name_to_gold_mat_dict.keys()) == set(
                self.model.train_output_tensors.keys()), \
                "Optimizer val_train_output_tensors and model train_output_tensors are out of sync"
        assert set(self.output_name_to_cost_gold_mat_dict.keys()) == set(
            self.model.gold_train_outputs_tensors.keys()), \
            "Optimizer train_output_tensors and model gold_output_tensors are out of sync! This also means the model" \
            " model train_output_tensors and model gold_output_tensors are out of sync!"
        if self.model.train_outputs_ordering is None:
            self.model.train_outputs_ordering = sorted(self.model.train_output_tensors.keys())
        else:
            ordering_set = set(self.model.train_outputs_ordering)
            assert all(output in ordering_set for output in
                       self.model.train_output_tensors.keys()), "train output ordering does not include all model " \
                                                                "outputs!"

    def sanity_check_train_inputs_ordering(self, validate=False):
        assert len(self.input_name_to_gold_mat_dict.keys()) > 0, "No train_inputs configured in " \
                                                                         "optimizer!"
        assert set(self.input_name_to_gold_mat_dict.keys()) == set(
            self.model.train_input_tensors.keys()), \
            "Optimizer train_input_tensors and model train_input_tensors are out of sync"
        if validate:
            assert set(self.val_input_name_to_gold_mat_dict.keys()) == set(self.model.train_input_tensors.keys()), \
                                    "Optimizer val_train_input_tensors and model train_input_tensors are out of sync!"


        if self.model.train_inputs_ordering is None:
            self.model.train_inputs_ordering = sorted(self.model.train_input_tensors.keys())
        else:
            ordering_set = set(self.model.train_inputs_ordering)
            assert all(input in ordering_set for input in
                       self.model.train_input_tensors.keys()), "train input ordering does not include all model " \
                                                                "inputs!"

    ####################################################################################################################
    ######################################### G E T     M E T H O D S ##################################################
    ####################################################################################################################

    def get_model_output_tensors(self):
        self.sanity_check_train_outputs_ordering()
        return [self.model.train_output_tensors[key] for key in self.model.train_outputs_ordering]

    def get_gold_output_tensors(self):
        self.sanity_check_train_outputs_ordering()
        return [self.model.gold_train_outputs_tensors[key] for key in self.model.train_outputs_ordering]

    def get_train_outputs(self): # Returns the gold train matrices for all outputs
        self.sanity_check_train_outputs_ordering()
        return [self.output_name_to_cost_gold_mat_dict[key][1] for key in self.model.train_outputs_ordering]

    def get_output_costs(self): # Returns the loss value for each model output with respect to its gold output
        self.sanity_check_train_outputs_ordering()
        return [self.output_name_to_cost_gold_mat_dict[key][0] for key in self.model.train_outputs_ordering]

    def get_model_input_tensors(self):
        self.sanity_check_train_inputs_ordering()
        return [self.model.train_input_tensors[key] for key in self.model.train_inputs_ordering]

    def get_train_inputs(self):
        self.sanity_check_train_inputs_ordering()
        return [self.input_name_to_gold_mat_dict[input] for input in self.model.train_inputs_ordering]

    def get_val_inputs(self):
        self.sanity_check_train_inputs_ordering(validate=True)
        return [self.val_input_name_to_gold_mat_dict[input] for input in self.model.train_inputs_ordering]

    def get_val_outputs(self):
        self.sanity_check_train_outputs_ordering(validate=True)
        return [self.val_output_name_to_gold_mat_dict[output] for output in self.model.train_outputs_ordering]

    def get_cost(self, is_train=True):
        # cost = T.sum(self.get_output_costs())
        # Note : We add all costs together so we can't track individual costs variations. Also, T.sum actually returns
        # a list with a single element rather than a scalar
        # scalar. This is not an issue when evaluating gradients since a scalar is depicted as an array with a single
        # element in theano.
        # cost = T.mean(self.get_output_costs()) # average the cost over all samples in the batch
        all_costs = self.get_output_costs()
        all_cost_names = [key for key in self.model.train_outputs_ordering]
        reg_cost, reg_cost_names = self.model.get_all_regularizer_costs(is_train=is_train)
        all_costs.extend(reg_cost)
        all_cost_names.extend(reg_cost_names)
        return all_costs, all_cost_names

    def get_scalar_cost(self,is_train=True):
        cost = T.sum(self.get_output_costs())
        # Note : We add all costs together so we can't track individual costs variations. Also, T.sum actually returns
        # a list with a single element rather than a scalar
        # scalar. This is not an issue when evaluating gradients since a scalar is depicted as an array with a single
        # element in theano.
        reg_costs, reg_cost_names = self.model.get_all_regularizer_costs(is_train=is_train)
        reg_cost = T.sum(reg_costs)
        return cost + reg_cost


    ####################################################################################################################
    ##################################### C O N F I G U R E     M E T H O D S ##########################################
    ####################################################################################################################

    def configure_costs_and_gold_outputs(self, output_name_to_cost_gold_mat_dict):
        """
        :param output_name_to_model_op_gold_op_gold_op_mat_cost_dict: Dictionary mapping an output name to the
               loss_type and train gold output matrix
        """
        output_names = set(output_name_to_cost_gold_mat_dict.keys())
        model_output_names = set(self.model.train_output_tensors.keys())
        assert output_names == model_output_names, "Mismatch between provided output configurations and model outputs" \
                                                   " tensors!"
        gold_output_names = set(self.model.gold_train_outputs_tensors.keys())
        assert output_names == gold_output_names, "Mismatch between provided output configurations and gold output " \
                                                  "tensors! This also implies mismatch between model output tensors " \
                                                  "and gold output tensors!"
        self.output_name_to_cost_gold_mat_dict = {}
        for key, val_tuple in output_name_to_cost_gold_mat_dict.items():
            assert len(val_tuple) == 2, "Must provide mapping from output_name to (cost, gold_mat)!"
            assert isinstance(val_tuple[0], str), "Specified cost type for " + key + " must be a string identifier " \
                                                  "for a cost type supported by Liebnetz!"
            assert isinstance(val_tuple[1], np.ndarray), "Specified gold mat for " + key + " must be a numpy.ndarray!"
            assert val_tuple[1].ndim == self.model.train_output_tensors[key].ndim, "No. of dimensions of provided " \
                    "gold output = " + str(val_tuple[1].ndim) + " for the output " + key + "doesn't match that " \
                    "of the corresponding model output tensor (=" + str(self.model.train_output_tensors[key].ndim) + ") !"
            assert val_tuple[1].ndim == self.model.gold_train_outputs_tensors[key].ndim, "No. of dims of provided " \
                   "gold output = " + str(val_tuple[1].ndim) + " for the output " + key + "doesn't match that " \
                   "of the corresponding gold output tensor (=" + str(self.model.gold_train_outputs_tensors.ndim) + ")!"
            # Checking for matching number of samples across all outputs
            # This can lead to false errors since all outputs need not have same number of samples:
            # eg. Predicting nouns for image sub regions but verbs for whole image
            # if len(self.output_name_to_cost_gold_mat_dict.keys()) > 0:
            #     assert self.output_name_to_cost_gold_mat_dict.values()[0][1].shape[0] == val_tuple[1].shape[0],\
            #         "No. of samples for " + key  + " doesn't match that of the other output mats!"
            # if len(self.input_name_to_gold_mat_dict.keys()) > 0:
            #     assert val_tuple[1].shape[0] == self.input_name_to_gold_mat_dict.values()[0].shape[0], "No. of " \
            #            "samples for output " + key + "doesn't match the number of input samples!"
            self.output_name_to_cost_gold_mat_dict[key] = (losses.get_loss(loss_type=val_tuple[0],
                                                                  y_pred=self.model.train_output_tensors[key],
                                                                  y_actual=self.model.gold_train_outputs_tensors[key]),
                                                           val_tuple[1])

    def configure_train_inputs(self, input_name_to_input_mat_dict):
        """

        :param input_tensor_to_input_dict: Dictionary mapping an input name to the gold input mat to be used for
                                           training
        """
        input_names = set(input_name_to_input_mat_dict.keys())
        model_input_names = set(self.model.train_input_tensors)
        assert input_names == model_input_names, "Mismatch between provide input configurations and model inputs!"
        for key, val in input_name_to_input_mat_dict.items():
            assert isinstance(val, np.ndarray), "Provided train input mat for " + key + " must be a np.ndarray!"
            assert val.ndim == self.model.train_input_tensors[key].ndim, "No. of dimensions of the provided input " \
                   "matrix = " + str(val.ndim) + " != no. od dims of corresponding model input tensor (=" + \
                   str(self.model.train_input_tensors[key].ndim) + ") !"
            # Checking for matching number of samples across all outputs
            # This can lead to false errors since all outputs need not have same number of samples:
            # eg. Predicting nouns for image sub regions but verbs for whole image
            # if len(self.output_name_to_cost_gold_mat_dict.keys()) > 0:
            #     assert self.output_name_to_cost_gold_mat_dict.values()[0][1].shape[0] == val.shape[0], \
            #         "No. of samples for input " + key + " doesn't match that of the output mats!"
            # if len(self.input_name_to_gold_mat_dict.keys()) > 0:
            #     assert val.shape[0] == self.input_name_to_gold_mat_dict.values()[0].shape[0], "No. of samples for" \
            #            " output " + key + "doesn't match the number of other input samples!"
        self.input_name_to_gold_mat_dict = {key:val for (key, val) in input_name_to_input_mat_dict.items()}

    def configure_val_inputs(self, val_input_name_to_input_mat_dict):
        """

        :param val_input_tensor_to_input_dict: Dictionary mapping an input name to the val input mat to be used for
                                               validation
        """
        input_names = set(val_input_name_to_input_mat_dict.keys())
        model_input_names = set(self.model.train_input_tensors)
        assert input_names == model_input_names, "Mismatch between provided val input configurations and model inputs!"
        for key, val in val_input_name_to_input_mat_dict.items():
            assert isinstance(val, np.ndarray), "Provided val input mat for " + key + " must be a np.ndarray!"
            assert self.input_name_to_gold_mat_dict[key].ndim == val.ndim, "No. of dimensions of val input for " \
                   "input " + key + " doesn't match that of its train input!"
        self.val_input_name_to_gold_mat_dict = {key: val for (key, val) in val_input_name_to_input_mat_dict.items()}

    def configure_val_outputs(self, val_output_name_to_gold_op_mat_dict):
        """
        :param val_output_name_to_gold_op_mat_dict: Dictionary mapping an output name to the model
               validation gold output matrix
        """
        output_names = set(val_output_name_to_gold_op_mat_dict.keys())
        model_output_names = set(self.model.train_output_tensors.keys())
        assert len(self.output_name_to_cost_gold_mat_dict.keys()) == len(model_output_names),\
            "Model outputs have not been configured yet. Please configure model outputs before configuring validation" \
            "outputs!"
        assert output_names == model_output_names, "Mismatch between provided output configurations and model outputs!"
        for key, val in val_output_name_to_gold_op_mat_dict.items():
            assert isinstance(val, np.ndarray), "Specified val output mat for " + key + " must be a numpy.ndarray!"
            assert self.output_name_to_cost_gold_mat_dict[key][1].ndim == val.ndim, "No. of dims of val gold output " \
                   "for output " + key + " doesn't match that of its train gold output"
        self.val_output_name_to_gold_mat_dict = {key: val for (key, val) in val_output_name_to_gold_op_mat_dict.items()}

    def print_costs(self, cost_names, cost_values):
        assert len(cost_names) == len(cost_values), "Mismatch between length of cost names and cost values!"
        for cost_index, cost_name in enumerate(cost_names):
            print("%s:%f\t\t"% (cost_name, cost_values[cost_index])),
            if (cost_index + 1) % 5 == 0:
                print("")
        print("")




class BatchGradientDescent(Optimizer):

    def __init__(self, model, clip_threshold=None):
        super(BatchGradientDescent, self).__init__(model=model, clip_threshold=clip_threshold)

    def get_updates(self):
        if self.updates is not None:
            return self.updates
        # train_cost, train_cost_names = self.get_cost(is_train=True)
        train_cost = self.get_scalar_cost(is_train=True)
        # if self.clip:
        #     train_cost = theano.gradient.grad_clip(x=train_cost, lower_bound=-1. * self.clip_threshold,
        #                                            upper_bound=1. * self.clip_threshold)
        #     # When computing the gradient, a clip with the specified lower and upper bound should be applied. This
        #     # does NOT clip the value of the cost itself. However, this method doesn't seem to work as expected, so
        #     # mode.get_all_gradients applies the clip manually for each trainable parameter of each trainable
        #     # component
        component_to_param_to_gradients = self.model.get_all_gradients(cost=train_cost,
                                                                       clip_threshold=self.clip_threshold)
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
        train_costs, train_cost_names = self.get_cost(is_train=True)
        print("Compiling train function...")
        self.train_model = theano.function(inputs=self.get_model_input_tensors()+self.get_gold_output_tensors(),
                                           outputs=self.get_model_output_tensors() + train_costs,
                                           updates=updates_dict)
        val_costs, val_cost_names = self.get_cost(is_train=False)
        if validate:
            print("CAUTION: Currently optimizer compiles validation function using computation graph for train. This "
                  "is not suitable for models that use dropout layers for example")
            self.validate_model = theano.function(inputs=self.get_model_input_tensors()+self.get_gold_output_tensors(),
                                                outputs=self.get_model_output_tensors() + val_costs)
        self.model.compile_model()
        return train_cost_names, val_cost_names

    def train(self, batch_size=None, n_epochs=100, compile_validation_fn=True, rnd=np.random.RandomState(), skip_compile=False,
              validate=False, suppress_train_reporting=False):
        eprint("Train function currently outputs the last output as the val/train loss at the end of each epoch. This"
              " isn't entirely general and should be fixed in future commits!")

        eprint("IT IS POSSIBLE TO SUPPORT BATCHING WITH MODELS DEFINED FOR SINGLE INSTANCES! CHANGE THE CHECKS FOR TENSOR"
              "DIMENSIONS IN THE SANITY CHECK AND CONFIG FUNCTIONS AT SOME POINT IN THE FUTURE!")
        if not skip_compile:
            train_cost_names, val_cost_names = self.compile_model(validate=compile_validation_fn)
            self.train_cost_names = train_cost_names
            self.val_cost_names = val_cost_names
        if self.train_cost_names is not None:
            train_cost_names = self.train_cost_names
        if self.val_cost_names is not None:
            val_cost_names = self.val_cost_names
        assert self.train_model is not None, "Train function not compiled!"
        if compile_validation_fn or validate:
            assert self.validate_model is not None, "Validation function not compiled!"
        train_inputs = self.get_train_inputs()
        train_outputs = self.get_train_outputs()
        input_size = train_inputs[0].shape[0]
        # We should not enforce that the number of samples be equal across different inputs and outputs
        # Eg. Predicting nouns/adjs for image subregions and predicting verbs globally
        # output_size = train_outputs[0].shape[0]
        # assert all(input.shape[0] == input_size for input in train_inputs), "Mismatch in number of input samples of" \
        #                                                                     " configured actual train_input_tensors"
        # assert all(output.shape[0] == output_size for output in train_outputs), "Mismatch in number of input samples of" \
        #                                                                         " configured actual train_output_tensors"
        # assert output_size == input_size, "Number of output samples and number of input samples don't match"
        if validate:
            validation_inputs = self.get_val_inputs()
            validation_outputs = self.get_val_outputs()
            # We should not enforce that the number of samples be equal across different inputs and outputs
            # val_input_size = validation_inputs[0].shape[0]
            # val_output_size = validation_outputs[0].shape[0]
            # assert all(val_input.shape[0] == val_input_size for val_input in validation_inputs),\
            #     "Mismatch in number of input samples of configured validation train_input_tensors"
            # assert all(val_output.shape[0] == val_output_size for val_output in validation_inputs), \
            #     "Mismatch in number of output samples of configured validation train_output_tensors"
            # assert val_input_size == val_output_size, "Validation input and validation output sizes don't match"
        if batch_size is not None:
            assert isinstance(batch_size, int), "Batch size is not an integer"
        assert isinstance(n_epochs, int), "Number of epochs must be an integer"
        train_index_shuffle_vect = np.array([i for i in range(input_size)])
        for epoch in range(n_epochs):
            if batch_size is not None:
                print("Epoch %d" % (epoch + 1))
                rnd.shuffle(train_index_shuffle_vect)
                tot_epoch_cost = [0. for cost_name in train_cost_names]
                tot_batches = 0.
                for batch_start_index in range(0, input_size, batch_size):
                    tot_batches += 1.
                    batch_indices = train_index_shuffle_vect[batch_start_index: batch_start_index + batch_size]
                    train_batch_inputs = [input[[batch_indices]] for input in train_inputs]
                    train_batch_outputs = [output[[batch_indices]] for output in train_outputs]
                    train_fn_outputs = self.train_model(*(train_batch_inputs + train_batch_outputs))
                    train_costs = train_fn_outputs[-len(train_cost_names):]
                    for cost_index, cost_val in enumerate(train_costs):
                        tot_epoch_cost[cost_index] += cost_val
                    # print("Training loss is : %f"%(train_fn_outputs[-1]))
                    # if validate:
                    #     val_fn_outputs = self.validate_model(*(validation_inputs+validation_outputs))
                    #     tot_epoch_val_cost += val_fn_outputs[-1]
                    #     print("Validation loss is : %f"%(val_fn_outputs[-1]))
                if not suppress_train_reporting:
                    print("Avg. batch costs for epoch %d"%(epoch+1))
                    train_costs = [train_cost/tot_batches for train_cost in tot_epoch_cost]
                    self.print_costs(cost_names=train_cost_names,cost_values=train_costs)
                    all_train_fn_outputs = self.validate_model(*(train_inputs + train_outputs))
                    all_train_costs = all_train_fn_outputs[-len(val_cost_names):]
                    print("Post epoch train costs after epoch %d:"%(epoch+1))
                    self.print_costs(cost_names=val_cost_names,cost_values=all_train_costs)
                if validate:
                    val_fn_outputs = self.validate_model(*(validation_inputs + validation_outputs))
                    # val_op_str = "\t".join([str(validation_outputs[i]) + " = " + str(val_fn_outputs[i])
                    #                         for i in range(len(validation_outputs))])
                    val_costs = val_fn_outputs[-len(val_cost_names):]
                    print("Post epoch val costs after_epoch %d:"%(epoch+1))
                    self.print_costs(cost_names=val_cost_names,cost_values=val_costs)
                    # print(val_op_str)
            else:
                train_fn_outputs = self.train_model(*(train_inputs + train_outputs))
                if not suppress_train_reporting:
                    train_costs = train_fn_outputs[-len(train_cost_names):]
                    # print("Post epoch training loss after epoch %d is"%(epoch+1))
                    print("Post epoch training loss after epoch is")
                    self.print_costs(cost_names=train_cost_names,cost_values=train_costs)
                if validate:
                    val_fn_outputs = self.validate_model(*(validation_inputs + validation_outputs))
                    # val_op_str = "\t".join([str(validation_outputs[i]) + " = " + str(val_fn_outputs[i])
                    #                         for i in range(len(validation_outputs))])
                    val_costs = val_fn_outputs[-len(val_cost_names):]
                    # print("Validation loss after epoch %d is" % (epoch + 1))
                    print("Validation loss after epoch is")
                    self.print_costs(cost_names=val_cost_names, cost_values=val_costs)
        eprint("Sometimes validation error doesn't match training error for the same data even without regularizers"
               "etc. INVESTIGATE!")
        # print("Done training!")
        return self.train_model, self.validate_model

    def validate(self, validation_inputs, validation_outputs):
        # validation_inputs = self.get_val_inputs()
        # validation_outputs = self.get_val_outputs()
        # assert len(validation_inputs) == len(self.input_name_to_tensor_gold_mat_dict.keys()), \
        #     "Inappropriate number of train_input_tensors provided"
        # assert len(validation_outputs) == len(self.output_name_to_model_op_gold_op_gold_mat_cost_dict.keys()), \
        #     "Inapprpriate number of train_output_tensors provided"
        # val_input_size = validation_inputs[0].shape[0]
        # val_output_size = validation_outputs[0].shape[0]
        # assert all(val_input.shape[0] == val_input_size for val_input in validation_inputs), \
        #     "Mismatch in number of input samples of configured validation train_input_tensors"
        # assert all(val_output.shape[0] == val_output_size for val_output in validation_inputs), \
        #     "Mismatch in number of output samples of configured validation train_output_tensors"
        # assert val_input_size == val_output_size, "Validation input and validation output sample sizes don't match"
        assert self.validate_model is not None, "Validate function has not been compiled yet!"
        val_fn_outputs = self.validate_model(*(validation_inputs + validation_outputs))
        print("Validation loss is : %f"%(val_fn_outputs[-1]))


    # MIGRATE FUNCTIONS FROM BatchGradientDescent TO Optimizer parent class as appropriate
    # COMPLETE THE VALIDATE FUNCTION!! ---> chekcing types of arguments, sufficiency of train_input_tensors etc.
    # Implement methods to clear out all data structures
    # Ensure order of train_output_tensors and train_input_tensors is constant
    # CONFIRM ALL OF THE BELOW HAVE BEEN SUITABLY CHANGED TO WORK WITH THE OPTIMIZER CLASS
