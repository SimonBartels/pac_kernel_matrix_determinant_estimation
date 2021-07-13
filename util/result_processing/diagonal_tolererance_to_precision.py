import numpy as np
import mlflow

from choleskies.openblas.pivoted_cholesky import PivotedCholeskyBLAS
from registry import CLIP
from result_management import ENV_CPUS, find_experiments_with_tags, ALGORITHM, SEED, get_run_list_from_dataframe
from util.result_processing.result_tables import StoppedAlgorithmsTable


def _get_precision(run_data_metrics):
    U = run_data_metrics[StoppedAlgorithmsTable.UPPER_BOUND]
    #m = runs[StoppedAlgorithmsTable.STEP]
    UL2 = run_data_metrics[StoppedAlgorithmsTable.LOG_DET_ESTIMATE]
    L = 2 * UL2 - U
    if np.sign(U) == np.sign(L):
        r = (U - L) / 2 / np.min([np.abs(U), np.abs(L)])
    else:
        r = 1.
    return r


def get_precisions_from_diagonal_tolerance(dataset, kfunc, sn2, clip=CLIP, seed=0, dt=None, exps=None, env=None):
    if exps is None:
        exps = find_experiments_with_tags(dataset=dataset, kernel=kfunc, environment=env, sn2=sn2, clip=clip)
        assert(len(exps) <= 1)
    sig = PivotedCholeskyBLAS().get_signature()
    filter_string = "tags." + ALGORITHM + "='" + sig + "' and tags." + SEED + "='%i'" % seed
    if dt is not None:
        filter_string += " and tags." + ALGORITHM + ".diagonal_tolerance='%s'" % (str(dt))
    runs = mlflow.tracking.MlflowClient().search_runs(
        [e.experiment_id for e in exps],
        filter_string=filter_string)
    return _get_precisions_from_pivoted_cholesky_runs_list(runs)


def _get_precisions_from_pivoted_cholesky_runs_list(runs: []):
    rs = []
    for run in runs:
        r = _get_precision(run.data.metrics)
        assert (r >= 0)
        rs.append(r)
    return rs


def _get_precisions_from_pivoted_cholesky_runs_df(runs):
    return _get_precisions_from_pivoted_cholesky_runs_list(get_run_list_from_dataframe(runs))
