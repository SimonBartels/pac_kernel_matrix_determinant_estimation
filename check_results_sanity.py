"""
Script to check whether all results exist.
"""
import warnings
import mlflow
import shutil
from os.path import sep

import numpy as np

from choleskies.openblas.default_cholesky import DefaultCholeskyBLAS
from choleskies.openblas.pivoted_cholesky import PivotedCholeskyBLAS
from choleskies.openblas.stopped_cholesky import StoppedCholeskyBLAS
from registry import KERNEL_DICT, CLIP
from result_management import ENV_CPUS, find_experiments_with_tags, delete_runs_if_crashed, filter_runs_with_tags, \
    get_run_list_from_dataframe
from util.execution.cluster import execute_job_array_on_slurm_cluster
from util.result_processing.diagonal_tolererance_to_precision import _get_precisions_from_pivoted_cholesky_runs_df

datasets = ['metro', 'tamilnadu_electricity', 'pm25', 'protein', 'bank']

sn2 = 1e-3
theta = 0.
seeds = [str(i) for i in range(0, 10)]
ds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
ls = [-1., 0., 1., 2.]
ks = list(KERNEL_DICT.keys())

cpus = 40
environment = {ENV_CPUS: cpus}
block_size = 256 * cpus


command_template = "python run_single_configuration.py -a %s %s -d %s -k %s -k-ls %s -s %i"
command_list = []


def delete_older_runs(run_df) -> []:
    run_df.sort_values(by=['start_time'])
    run_ls = get_run_list_from_dataframe(run_df)
    start_time = run_ls[0].info.start_time
    for i in range(1, len(run_ls)):
        run = run_ls[i]
        # make sure we keep the most recent run
        assert(run.info.start_time < start_time)
        remove = mlflow.get_tracking_uri() + sep + run.info.experiment_id + sep + run.info.run_id
        #print(remove)
        shutil.rmtree(remove)


def check_consistency(dataset, kernel, algorithm, algo_parameters, seed, runs):
    runs = filter_runs_with_tags(runs, algorithm, algo_parameters, seed)
    remaining_runs = delete_runs_if_crashed(get_run_list_from_dataframe(runs))

    if len(runs) != len(remaining_runs):
        runs = mlflow.search_runs(experiment_ids=[remaining_runs[0].info.experiment_id])
        runs = filter_runs_with_tags(runs, algorithm, algo_parameters, seed)

    if len(remaining_runs) > 1:
        warnings.warn("More than one result for configuration: ")
        delete_older_runs(runs)
        runs = mlflow.search_runs(experiment_ids=[remaining_runs[0].info.experiment_id])
        runs = filter_runs_with_tags(runs, algorithm, algo_parameters, seed)

    elif len(remaining_runs) == 0:
        parameters = ""
        sig = "--" + algorithm.get_signature()
        for k in algo_parameters.keys():
            parameters += sig + "-" + k.replace('_', '-') + " " + str(algo_parameters[k]) + " "
        command = command_template % (algorithm.get_signature(), parameters, dataset, kernel.name,
                                      str(kernel.parameters["ls"]), int(seed))
        print(command)
        command_list.append(command)
    return runs


for dataset in datasets:
    for l in ls:
        for k in ks:
            print(dataset + " " + k + " " + str(l))
            kernel = KERNEL_DICT[k]
            kernel.initialize(D=None, var=theta, ls=l)

            exps = find_experiments_with_tags(dataset, kernel, environment, sn2, CLIP)
            assert(len(exps) == 1)
            runs = mlflow.search_runs(experiment_ids=[e.experiment_id for e in exps])
            for s in seeds:
                check_consistency(dataset=dataset, kernel=kernel, algorithm=DefaultCholeskyBLAS(), algo_parameters={},
                                  seed=s, runs=runs)
                for dt in ds:
                    piv_runs = check_consistency(dataset=dataset, kernel=kernel, algorithm=PivotedCholeskyBLAS(),
                                      algo_parameters={"diagonal_tolerance": str(dt)}, seed=s, runs=runs)
                    rs = _get_precisions_from_pivoted_cholesky_runs_df(piv_runs)
                    assert(len(rs) <= 1)
                    if len(rs) == 0:
                        # a pivoted run is still missing
                        # we have to run that first before we can run the stopped experiment
                        continue
                    #r = str(0.0) if rs[0] == 0 else ("%.6f" % rs[0]).rstrip('0')
                    # convert precision row to numerical value---hell, mlflow can be treacherous...
                    stopped_runs = filter_runs_with_tags(runs, StoppedCholeskyBLAS(),
                                                         algo_parameters={"delta": str(0.1)}, seed=s)
                    stopped_runs = stopped_runs.astype({"tags.algorithm.r": float})
                    stopped_runs = stopped_runs.round({"tags.algorithm.r": 6})
                    r = np.around(rs[0], decimals=6)
                    check_consistency(dataset=dataset, kernel=kernel, algorithm=StoppedCholeskyBLAS(),
                                      algo_parameters={"r": r, "delta": str(0.1)}, seed=s, runs=stopped_runs)
if len(command_list) > 0:
    print(execute_job_array_on_slurm_cluster(commands=command_list, cpus=cpus))
else:
    print("All results exist.")
