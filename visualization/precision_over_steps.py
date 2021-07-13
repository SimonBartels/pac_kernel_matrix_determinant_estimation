import warnings

import matplotlib.pyplot as plt
import mlflow
from matplotlib.pyplot import plot, text, xlabel, ylabel, xlim, ylim, title
from pickle import load

from choleskies.openblas.default_cholesky import DefaultCholeskyBLAS
from choleskies.openblas.pivoted_cholesky import PivotedCholeskyBLAS
from data_sets.load_dataset import load_dataset, latex_name
from kernel.get_kernel_properties import get_upper_and_lower_bounds
from registry import KERNEL_DICT
from result_management import ALGORITHM, KERNEL, DATASET, SN2, SEED
from util.result_processing.diagonal_tolererance_to_precision import _get_precisions_from_pivoted_cholesky_runs_df
from util.result_processing.result_tables import DefaultCholeskyTable, DefaultTable, StoppedAlgorithmsTable
from util.visualization.visualization_utils import get_displayed_length_scale
from visualization.speed_gain import _make_legend
from visualization.visualization_constants import *
from util.stop_cond import compute_bounds
from kernel.isotropic_kernel import IsotropicKernel


def get_dataset_and_kernel_properties(k: IsotropicKernel, theta: float, ls: float, dataset: str, sn2: float) \
        -> (float, float, int, int):
    X = load_dataset(dataset)
    N, D = X.shape
    assert(not hasattr(k, 'params'))  # the kernel should not be already initialized!
    k.initialize(D, theta, ls)
    Cp, Cm = get_upper_and_lower_bounds(k, sn2, X)
    return Cp, Cm, N, D


def make_bound_legend():
    gca = _make_legend()
    gca.plot(-1, 0, diagonal_precision_style, color='black', label="\\DiagonalPrecision{}")
    gca.plot(-1, 0, color=ora, label="\\overhead{}")
    plt.legend(loc="center", ncol=4)


def make_plot(experiment_name: str, delta: str) -> (np.ndarray, np.ndarray):
    gca = plt.gca()
    mlflow.set_experiment(experiment_name)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    k = KERNEL_DICT[experiment.tags[KERNEL]]
    theta = float(experiment.tags[KERNEL + ".var"])
    ls = float(experiment.tags[KERNEL + ".ls"])
    dataset = experiment.tags[DATASET]
    sn2 = float(experiment.tags[SN2])
    Cp, Cm, N, D = get_dataset_and_kernel_properties(k, theta, ls, dataset, sn2)

    runs = mlflow.search_runs()

    seeds = sorted(set(runs["tags." + SEED]))

    ds = set(runs["tags." + ALGORITHM + ".diagonal_tolerance"])
    ds.discard(None)
    rs = {}

    l_label = "\\StoppedChol{}"
    for i, seed in enumerate(seeds):
        run = runs.loc[(runs["tags." + ALGORITHM] == DefaultCholeskyBLAS().get_signature()) &
                       (runs["tags." + SEED] == seed)]
        assert(run.shape[0] == 1)
        f = open(run["tags." + DefaultCholeskyTable.DIAGONAL].item(), "rb")
        l = load(f)
        mus, LBd, UBd, stop_cond_det, UB, stop_cond_r = compute_bounds(N, N, l, Cm, Cp, float(delta))
        plot_second_stopping_condition(N, stop_cond_det, stop_cond_r, label=l_label)
        l_label = None

    x_label = "\\PivotedChol{}"
    for d in ds:
        piv_runs = runs.loc[(runs["tags." + ALGORITHM] == PivotedCholeskyBLAS().get_signature()) &
                            (runs["tags." + ALGORITHM + ".diagonal_tolerance"] == d)]
        rs_ = _get_precisions_from_pivoted_cholesky_runs_df(piv_runs.sort_values(by=['tags.' + SEED]))
        mean_r = np.mean(rs_)
        assert(np.all(np.array(rs_) > 1.) or np.var(rs_) < 1e-2)
        rs[float(d)] = mean_r

        for i, run_id in enumerate(piv_runs["run_id"]):
            hist = mlflow.tracking.MlflowClient().get_metric_history(run_id=run_id,
                                                                     key=DefaultTable.CUM_TIME)
            assert(len(hist) == 1)
            stopped_at = hist[0].step
            run_data_metrics = mlflow.get_run(run_id).data.metrics
            U = run_data_metrics[StoppedAlgorithmsTable.UPPER_BOUND]
            UL2 = run_data_metrics[StoppedAlgorithmsTable.LOG_DET_ESTIMATE]
            L = 2 * UL2 - U
            r = (U - L) / 2 / np.min([np.abs(U), np.abs(L)])
            plot(stopped_at, float(r), 'x', color=dre, label=x_label)
            x_label = None
    setup_stop_cond_plot(dataset, N, D, k.name, theta, ls, delta, rs, [float(d) for d in ds], gca)


def setup_stop_cond_plot(dataset: str, N: int, D: int, k: str, theta: float, l: float, delta: str, rs: {}, ds: [float], gca) -> ():
    plt.yscale('log')
    ylim([lowest_r, 1e4])
    xlim(0, N)
    title('stopping condition')
    xlabel('\\labelx{}')
    l_disp = get_displayed_length_scale(l)
    text(gca.get_xlim()[1] / 16 * 15, 1000, '\\runinfo{%s}{%i}{%i}{%s}{%f}{%0.2f}{%s}' % (latex_name(dataset), D, N, k, theta, l_disp, delta),
         horizontalalignment='right', verticalalignment='bottom', fontsize='x-large')

    label = diagonal_precision_label
    for d in ds:
        r = rs[d]
        if r < lowest_r:
            continue
        plot(np.arange(0, N), r * np.ones(N), '--', color=diagonal_precision_color, label=label)
        label = None

        verticalalignment = 'bottom'  # default
        if d == 0.005 or d == 0.05:
            verticalalignment = 'top'

        if d < 0.05:
            text(0, r, '\\$d=%s\\$' % str(d), color=diagonal_precision_color, horizontalalignment='left',
                 verticalalignment=verticalalignment)
        else:
            text(N, r, '\\$d=%s\\$' % str(d), color=diagonal_precision_color, horizontalalignment='right',
                 verticalalignment=verticalalignment)
    ylabel('\\labely{}')


def plot_second_stopping_condition(N, stop_cond_det, stop_cond_r, label=None):
    plot(np.arange(0, np.size(stop_cond_r)), stop_cond_r, color=mpg, label=label)


def plot_bounds(lDets, N, Cm, Cp, UBd, LBd, UB):
    title('bounds and final log determinant')
    # plot final determinant only if computation was successful to the end
    if np.size(LBd) == N:
        plot(np.arange(0, N), lDets[-1] * np.ones(N), color=ora)
    plot(np.arange(0, N), LBd, color=dre)
    plot(np.arange(0, N), UBd, color=dre)
    plot(np.arange(0, N), UB, color=mpg)
