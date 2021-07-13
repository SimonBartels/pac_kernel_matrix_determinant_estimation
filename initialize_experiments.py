import mlflow

from parser import get_default_parser
from registry import KERNEL_DICT, ENVIRONMENT_DICT, CLIP
from result_management import get_results_path, make_experiment_name


def initialize_experiment(dataset, sn2, kernel_name, theta, l, clip=CLIP):
    k = KERNEL_DICT[kernel_name].initialize(D=None, var=theta, ls=l)
    assert(mlflow.get_tracking_uri() == get_results_path())
    experiment_name = make_experiment_name(dataset, k, ENVIRONMENT_DICT, sn2, clip)
    mlflow.set_experiment(experiment_name)
    # we must have defined the run first, to have an active run object...
    #set_experiment_tags(dataset, k, ENVIRONMENT_DICT, sn2)


def main(**args):
    dataset = args["dataset"]
    sn2 = args["sn2"]
    l = args["kernel_ls"]
    theta = args["kernel_var"]
    kernel_name = args["kernel"]
    clip = args["clip"]
    initialize_experiment(dataset=dataset, sn2=sn2, kernel_name=kernel_name, theta=theta, l=l, clip=clip)


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    main(**vars(args))
