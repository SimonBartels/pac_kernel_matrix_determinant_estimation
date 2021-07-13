"""
script for running all stopped Cholesky experiments
"""
import mlflow

from choleskies.openblas.pivoted_cholesky import PivotedCholeskyBLAS
from choleskies.openblas.stopped_cholesky import StoppedCholeskyBLAS
from registry import KERNEL_DICT, CLIP
from result_management import get_results_path, ENV_CPUS, find_experiments_with_tags, filter_runs_with_tags
from util.execution.cluster import execute_job_array_on_slurm_cluster
from util.result_processing.diagonal_tolererance_to_precision import _get_precisions_from_pivoted_cholesky_runs_df

assert(mlflow.get_tracking_uri() == get_results_path())

cpus = 40

seeds = range(0, 10)
datasets = ['metro', 'tamilnadu_electricity', 'pm25', 'protein', 'bank']
ls = [-1., 0., 1., 2.]
deltas = [0.1]

r_deltas = [(0, 0.1)]  # measure overhead
ds = [0.005, 0.01, 0.05, 0.1, 0.5]  # diagonal tolerances

sn2 = 1e-3
theta = 0.

max_iterations = 60000


algo_parameters = StoppedCholeskyBLAS().get_default_parameter_dictionary()
env_dict = {ENV_CPUS: cpus}
commands = []
for l in ls:
    for k in list(KERNEL_DICT.keys()):
        kfunc = KERNEL_DICT[k]
        kfunc.initialize(None, theta, l)
        for dataset in datasets:
            exps = find_experiments_with_tags(dataset, kfunc, env_dict, sn2, CLIP)
            assert (len(exps) == 1)
            runs = mlflow.search_runs(experiment_ids=[exps[0].experiment_id])
            for seed in seeds:
                r_deltas_ = r_deltas.copy()
                for d in ds:
                    piv_runs = filter_runs_with_tags(runs, PivotedCholeskyBLAS(), {"diagonal_tolerance": str(d)}, seed)
                    rs = _get_precisions_from_pivoted_cholesky_runs_df(piv_runs)
                    assert(len(rs) <= 1)  # maybe we do not have all results yet--we'll notice in the consistency check
                    if len(rs) == 1 and rs[0] > 0:
                        r_deltas_.extend([(rs[0], delta) for delta in deltas])
                for r, delta in r_deltas_:
                    command = "python run_single_configuration.py -a scb -mi %i" % max_iterations
                    command += " -scb-r %f -scb-d %f" % (r, delta)
                    command += " -d %s" % dataset
                    command += " -k %s -k-ls %f -k-var %f" % (k, l, theta)
                    command += " -sn2 %f" % sn2
                    command += " -s %i" % seed
                    commands.append(command)
execute_job_array_on_slurm_cluster(commands, cpus=cpus)
