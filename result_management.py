import os
import warnings
from os.path import sep
import mlflow
from warnings import warn
import pickle
import numpy as np
from mlflow.entities import ViewType
from time import time
import shutil

from data_sets.load_dataset import load_dataset
from kernel.isotropic_kernel import IsotropicKernel
from util.result_processing.result_tables import DefaultTable, DefaultCholeskyTable


def get_results_path():
    #return os.path.join(os.path.join(os.getcwd(), "results"), "mlruns")
    return os.path.join(os.path.join(os.path.dirname(__file__), "results"), "mlruns")


mlflow.set_tracking_uri(get_results_path())


ALGORITHM = "algorithm"
KERNEL = "kernel"
DATASET = "dataset"
ENVIRONMENT = "env"
ENV_CPUS = "cpus"
ENV_PROC = "processor"
SEED = "seed"
SN2 = "sn2"
NODE_NAME = "node_name"
CLIP = "clip"


def apply_binary_operator_to_experiment(f, x, dataset, kernel, environment, sn2, clip):
    x = f(x, DATASET, dataset)
    x = f(x, KERNEL, kernel.name)
    for key in kernel.parameters:
        x = f(x, KERNEL + '.' + key, kernel.parameters[key])
    for key in environment:
        x = f(x, ENVIRONMENT + '.' + key, environment[key])
    x = f(x, SN2, sn2)
    x = f(x, CLIP, clip)
    return x


def apply_binary_operator_to_run(f, x, dataset, kernel, algorithm, algorithm_parameters, environment, sn2, seed, clip):
    x = apply_binary_operator_to_experiment(f, x, dataset, kernel, environment, sn2, clip)
    x = apply_binary_operator_to_run_only(f, x, algorithm, algorithm_parameters, seed)
    return x


def apply_binary_operator_to_run_only(f, x, algorithm, algorithm_parameters, seed):
    x = f(x, ALGORITHM, algorithm.get_signature())
    for key in algorithm_parameters:
        x = f(x, ALGORITHM + '.' + key, algorithm_parameters[key])
    x = f(x, SEED, seed)
    return x


def build_filter_string(*args):
    def f(s, a, b):
        return s + "tag." + a + '="' + str(b) + '" and '
    return apply_binary_operator_to_run(f, "", *args)


def make_experiment_name(*args):
    def f(s, a, b):
        return s+str(a)+str(b)
    return apply_binary_operator_to_experiment(f, "", *args)


def get_initialized_kernel_from_experiment(e: mlflow.entities.experiment) -> IsotropicKernel:
    from registry import KERNEL_DICT
    k = KERNEL_DICT[e.tags[KERNEL]]
    params = {}
    prefix_len = len(KERNEL) + 1  # plus one for the dot
    for t in e.tags:
        if t.startswith(KERNEL):
            if t == KERNEL:
                continue
            params[t[prefix_len:]] = float(e.tags[t])
    k.initialize(None, **params)
    return k


def get_dataset_properties_from_experiment(e: mlflow.entities.experiment):
    dataset = e.tags[DATASET]
    X = load_dataset(dataset)
    N, D = X.shape
    return dataset, N, D


def get_environment_dict_from_experiment(e: mlflow.entities.experiment):
    return {ENV_CPUS: e.tags[ENVIRONMENT + '.' + ENV_CPUS], ENV_PROC: e.tags[ENVIRONMENT + '.' + ENV_PROC]}


def experiment_name_is_consistent(e: mlflow.entities.experiment):
    k = get_initialized_kernel_from_experiment(e)
    env_dict = get_environment_dict_from_experiment(e)
    return e.name == make_experiment_name(e.tags[DATASET], k, env_dict, e.tags[SN2], e.tags[CLIP])


def _has_tag(e, last_check, tag, attribute):
    if not last_check:
        # if the last check failed, we do not need to bother checking further
        return False
    try:
        return e.tags[tag] == str(attribute)
    except KeyError:
        warn("Tag " + tag + " caused an error for experiment with id " + e.experiment_id)
        return False


def find_experiments_with_tags(dataset, kernel, environment, sn2, clip):
    exps = mlflow.tracking.MlflowClient().list_experiments(view_type=ViewType.ALL)
    exps = [e for e in exps if apply_binary_operator_to_experiment(
        lambda *args: _has_tag(e, *args), True, dataset, kernel, environment, sn2, clip)]
    return exps


def find_runs_with_tags(dataset, kernel, environment, sn2, algorithm, algo_parameters, seed, clip, exps=None):
    if exps is None:
        #exp = mlflow.get_experiment_by_name(make_experiment_name(dataset, kernel, environment, sn2, clip))
        exps = find_experiments_with_tags(dataset, kernel, environment, sn2, clip)

    def _build_filter_string(*args):
        def f(s, a, b):
            return s + "tag." + a + '="' + str(b) + '" and '
        return apply_binary_operator_to_run_only(f, "", *args)

    runs = mlflow.tracking.MlflowClient().search_runs(
        experiment_ids=[exp.experiment_id for exp in exps],
        filter_string=_build_filter_string(algorithm, algo_parameters, seed),
        run_view_type=ViewType.ALL)
    return runs


def filter_runs_with_tags(runs, algorithm, algo_parameters, seed):
    def _build_filter_string(*args):
        def f(s, a, b):
            return s & (runs["tags." + a] == b)
        return apply_binary_operator_to_run_only(f, True, *args)
    return runs.loc[_build_filter_string(algorithm, algo_parameters, seed)]


def get_steps_from_config(dataset, kernel, environment, sn2, algorithm, algo_parameters, seed, clip):
    runs = find_runs_with_tags(dataset, kernel, environment, sn2, algorithm, algo_parameters, seed, clip)
    assert(len(runs) <= 1)
    if len(runs) == 0:
        return []
    if runs.shape[1] == 0:
        return []
    #results = mlflow.tracking.MlflowClient().get_metric_history(runs['run_id'][0], DefaultTable.LOG_DET_ESTIMATE)
    results = mlflow.tracking.MlflowClient().get_metric_history(runs[0].info.run_id, DefaultTable.LOG_DET_ESTIMATE)
    return [r.step for r in results]


def save_diagonal(K):
    e = mlflow.get_experiment(mlflow.active_run().info.experiment_id)
    tags = e.tags
    file_name = '__'.join([k + '_' + str(tags[k]) for k in tags.keys() if k.startswith(DATASET) or k.startswith(KERNEL)]) \
                + str(time()) + '.d'
    path = os.path.join(os.path.join("results", "diagonals"), file_name)
    with open(path, "wb+") as f:
        pickle.dump(np.diag(K), f)
        f.flush()
        mlflow.set_tag(DefaultCholeskyTable.DIAGONAL, path)


def delete_runs_if_crashed(run_ls: []) -> []:
    deleted_runs = []
    for run in run_ls:
        if not DefaultTable.LOG_DET_ESTIMATE in run.data.metrics:
            remove = mlflow.get_tracking_uri() + sep + run.info.experiment_id + sep + run.info.run_id
            shutil.rmtree(remove)
            deleted_runs.append(run)
    remaining = set(run_ls).difference(set(deleted_runs))
    return list(remaining)


def get_run_list_from_dataframe(runs):
    run_ls = []
    for run_id in runs["run_id"].values:
        try:
            run_ls.append(mlflow.tracking.MlflowClient().get_run(run_id))
        except:
            warnings.warn("Could not find run with id %s" % run_id)
    return run_ls
