import numpy as np
import theano
from numpy.random import RandomState
import initializers

def initializers_selector_check():
    for init_type in ["uniform_param_init", "normal_param_init", "lecun_uniform_param_init", "glorot_uniform_param_init",
                  "glorot_normal_param_init", "he_uniform_param_init", "he_normal_param_init",
                  "orthogonal_param_init", "identity_param_init", "zero_param_init", "one_param_init"]:
        args_dict = {"name": init_type + "_dummy", "dim_tuple": (3,3), "rnd": np.random}
        print(init_type)
        init_value = initializers.get_init_value(init_type, **args_dict)
        print(init_value.name)
        print(init_value.get_value())
        raw_input("Enter to continue")

def seed_set_check():
    rnd = RandomState(0)
    init_type = "uniform_param_init"
    args_dict = {"name": init_type + "_dummy", "dim_tuple": (3,3), "rnd": rnd}
    print(init_type)
    init_value = initializers.get_init_value(init_type, **args_dict)
    print(init_value.name)
    print(init_value.get_value())

    rnd = RandomState(0)
    rnd2 = RandomState(10)
    args_dict = {"name": init_type + "_dummy2", "dim_tuple": (3, 3), "rnd": rnd}
    print(init_type)
    init_value = initializers.get_init_value(init_type, **args_dict)
    print(init_value.name)
    print(init_value.get_value())

    args_dict = {"name": init_type + "_dummy2", "dim_tuple": (3, 3), "rnd": rnd2}
    print(init_type)
    init_value = initializers.get_init_value(init_type, **args_dict)
    print(init_value.name)
    print(init_value.get_value())


initializers_selector_check()
seed_set_check()