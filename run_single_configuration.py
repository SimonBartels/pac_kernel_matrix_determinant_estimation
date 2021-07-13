"""
script to execute a single configuration
"""
import os
import gc
import numpy as np
import random
import mlflow

import registry
from parser import get_default_parser
from registry import ALGORITHM_DICT, KERNEL_DICT, ENVIRONMENT_DICT
from data_sets.load_dataset import load_dataset
from result_management import ALGORITHM, build_filter_string, apply_binary_operator_to_run, \
    apply_binary_operator_to_experiment, get_results_path, make_experiment_name, NODE_NAME, \
    apply_binary_operator_to_run_only, find_runs_with_tags
from util.setup_kernel_matrix import setup_kernel_matrix


def define_run(run: mlflow.run, *args):
    """
    Adds all necessary tags to the mlflow run.
    """
    def f(s, a, b):
        #run.set_param(a, b)
        mlflow.set_tag(a, b)
    apply_binary_operator_to_run_only(f, [], *args)
    mlflow.set_tag(NODE_NAME, os.uname()[1])


def set_experiment_tags(*args):
    """
    Sets the tags for the experiment.
    """
    set_experiment_tags_for(mlflow.active_run().info.experiment_id, *args)


def set_experiment_tags_for(experiment_id, *args):
    """
    Sets tags for a given experiment.
    """
    def f(x, a, b):
        mlflow.tracking.MlflowClient().set_experiment_tag(experiment_id, a, b)
    apply_binary_operator_to_experiment(f, [], *args)


def main(**args):
    verbose = args["verbose"]
    registry.VERBOSE = verbose

    # set parameters
    seed = args["seed"]
    dataset = args["dataset"]
    sn2 = args["sn2"]

    l = args["kernel_ls"]
    theta = args["kernel_var"]

    # set seeds
    np.random.seed(seed)
    random.seed(seed)

    # prepare variables
    X = load_dataset(dataset)
    if verbose:
        print("dataset loaded")
    k = KERNEL_DICT[args["kernel"]].initialize(X.shape[1], var=theta, ls=l)

    algorithm = ALGORITHM_DICT[args[ALGORITHM]]
    parameters = algorithm.get_parameter_dictionary(args)

    clip = args["clip"]
    K = setup_kernel_matrix(k, sn2, X, clip=clip)
    if verbose:
        print("kernel matrix built")
    algorithm.set_environment(sn2, k)
    algorithm.init(**parameters)
    assert(mlflow.get_tracking_uri() == get_results_path())
    experiment_name = args["experiment_name"]
    if experiment_name is None:
        experiment_name = make_experiment_name(dataset, k, ENVIRONMENT_DICT, sn2, clip)
    mlflow.set_experiment(experiment_name)

    run = mlflow.start_run()
    define_run(run, algorithm, parameters, seed)

    # we must have defined the run first, to have an active run object...
    set_experiment_tags(dataset, k, ENVIRONMENT_DICT, sn2, clip)

    if verbose:
        print("running configuration")
    gc.collect()
    gc.disable()
    algorithm.run_configuration(K, max_iterations=args["max_iterations"], max_time=args["max_time"])
    mlflow.end_run()


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    main(**vars(args))
