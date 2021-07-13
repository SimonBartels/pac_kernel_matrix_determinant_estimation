import warnings

import matplotlib.pyplot as plt
import mlflow
from matplotlib.pyplot import text

from choleskies.openblas.default_cholesky import DefaultCholeskyBLAS
from choleskies.openblas.pivoted_cholesky import PivotedCholeskyBLAS
from choleskies.openblas.stopped_cholesky import StoppedCholeskyBLAS
from data_sets.load_dataset import latex_name
from result_management import ALGORITHM, SEED, \
    get_initialized_kernel_from_experiment, get_dataset_properties_from_experiment
from util.result_processing.diagonal_tolererance_to_precision import _get_precisions_from_pivoted_cholesky_runs_df
from util.result_processing.result_tables import DefaultTable
from util.visualization.visualization_utils import get_displayed_length_scale, _rs_to_str
from visualization.visualization_constants import *
from util.result_processing.performance_metrics import pm_speed_up, pm_fraction


piv_run_color = dre
piv_run_symbol = 'x'
stopped_run_color = mpg
stopped_run_symbol = 'o'
ste_color = np.zeros(3)


def _add_axis_labels(gca):
    gca.set_ylabel("\\labely{}") #, loc='top')
    gca.set_xlabel("\\labelx{}")


def _make_legend():
    fig = plt.figure()
    gca = fig.add_axes([0, 0, 1, 1], frameon=False)
    stopped_label = StoppedCholeskyBLAS().get_latex_name()
    gca.plot(-1, 0, stopped_run_symbol, color=stopped_run_color, mfc='none', label=stopped_label)
    piv_label = PivotedCholeskyBLAS().get_latex_name()
    gca.plot(-1, 0, piv_run_symbol, color=piv_run_color, mfc='none', label=piv_label)
    gca.set_xlim([0, 1])
    gca.set_xticks([])
    gca.set_yticks([])
    return gca


def make_legend():
    gca = _make_legend()
    gca.plot(-1, 0, 's', color=blu, mfc='none', clip_on=False, label="\\DefaultChol{}")
    gca.plot(-1, 0, color=ora, label="\\overhead{}")
    plt.legend(loc="center", ncol=4)


def make_inverse_plot(experiment_name: str, delta: str, gca):
    gca.set_xlim([0, 2.5])
    ts = _make_plot(experiment_name, delta, gca, performance_metric=pm_fraction, invert_axis=True)
    plt.yscale('log')
    lowest_r = 1e-4
    plt.ylim([lowest_r, 1e4])
    lims = gca.get_ylim()
    gca.plot([1.05, 1.05], [lims[0] + 1e-5, lims[1] - 1e-5], color=ora, label="\\overhead{}")
    gca.set_ylim(lims)  # otherwise the plot would expand
    gca.text(gca.get_xlim()[1] / 16 * 15, 1000, "\\meantime{%i}" % int(np.mean(ts)), fontsize='x-large',
             horizontalalignment='right', verticalalignment='bottom')
    _add_axis_labels(gca)
    gca.yaxis.tick_right()


def make_tight_joint_plot(experiment_name: str, delta :str, gca):
    ts = _make_plot(experiment_name, delta, gca, performance_metric=pm_fraction)
    mts = np.mean(ts)
    lims = gca.get_xlim()
    gca.plot([lims[0] + 1e-5, lims[1] - 1e-5], [1.05, 1.05], color=ora, label="\\overhead{}")
    gca.plot(lims[0] * np.ones(ts.shape) + 1e-5, ts / mts, 's', color='black', mfc='none', clip_on=False, label="\\DefaultChol{}")
    gca.set_xlim(lims)  # otherwise the plot would expand
    gca.set_ylim([0, 2.5])
    yticks = gca.get_yticks()
    ids, = np.where(yticks == 1)
    ytick_labels = [str(y) for y in yticks]
    ytick_labels[ids[0]] = "(" + str(int(np.mean(ts))) + " s) 1"
    gca.set_yticklabels(ytick_labels)
    _add_axis_labels(gca)


def _make_plot(experiment_name: str, delta: str, gca=None, performance_metric=pm_speed_up, invert_axis=False) -> (np.ndarray, np.ndarray):
    if gca is None:
        gca = plt.gca()

    mlflow.set_experiment(experiment_name)
    runs = mlflow.search_runs()

    seeds, ds = get_parameters_from_runs(runs)

    e = mlflow.get_experiment_by_name(experiment_name)
    dataset, N, D = get_dataset_properties_from_experiment(e)
    k = get_initialized_kernel_from_experiment(e)

    setup_plot(dataset, N, D, k.name, k.parameters["var"], k.parameters["ls"], delta, ds, gca)

    # runs with default Cholesky
    def_runs = runs.loc[(runs["tags." + ALGORITHM] == DefaultCholeskyBLAS().get_signature())]
    ground_truth = def_runs["metrics." + DefaultTable.LOG_DET_ESTIMATE].to_numpy()
    assert(np.std(ground_truth) < 1e-7)
    ground_truth = np.mean(ground_truth)

    def assert_runs_deliver_precisions(runs, rs):
        estimates = runs["metrics." + DefaultTable.LOG_DET_ESTIMATE].to_numpy()
        assert(np.all(np.abs((estimates - ground_truth) / ground_truth) <= np.array(rs) + 1e-8))

    # times of the default Cholesky
    ts = def_runs["metrics." + DefaultTable.CUM_TIME].to_numpy()

    piv_label = PivotedCholeskyBLAS().get_latex_name()
    stopped_label = StoppedCholeskyBLAS().get_latex_name()
    diag_precision_label = diagonal_precision_label

    x_labels = []
    id = 1
    for d in ds:
        piv_runs = runs.loc[(runs["tags." + ALGORITHM] == PivotedCholeskyBLAS().get_signature()) &
                            (runs["tags." + ALGORITHM + ".diagonal_tolerance"] == str(d))]
        rs = _get_precisions_from_pivoted_cholesky_runs_df(piv_runs.sort_values(by=['tags.' + SEED]))
        mean_r = np.mean(rs)
        assert(np.all(np.array(rs) > 1.) or np.var(rs) < 1e-1)
        piv_values = piv_runs["metrics." + DefaultTable.CUM_TIME].to_numpy()
        plot_algo(gca, piv_values, id, performance_metric, mean_r, ts, symbol=piv_run_symbol, color=piv_run_color,
                  label=piv_label, invert_axis=invert_axis)
        piv_label = None
        x_labels.append(str(d) + '\\\\%.2f' % float(mean_r))

        srun_ = runs.loc[(runs["tags." + ALGORITHM] == StoppedCholeskyBLAS().get_signature()) &
                        (runs["tags." + ALGORITHM + ".delta"] == delta)]
        srun_ = srun_.astype({"tags.algorithm.r": float})
        srun_ = srun_.round({"tags.algorithm.r": 6})
        rs = np.around(rs, decimals=6)

        srun_times = []
        # because mlflow stores floats as strings, we have to search for the corresponding run...
        for i, seed in enumerate(seeds):
            r = rs[int(seed)]  # Ugh, that's somewhat of a dirty hack...
            srun = srun_.loc[(srun_["tags." + SEED] == seed) & (srun_["tags." + ALGORITHM + ".r"] == r)]
            if len(srun) == 0:
                # If there are no runs, then maybe we are trying to process old results.
                # In exploratory experiments, I considered the relative precision only of the first run.
                # The effect is negligible, though...
                warnings.warn("Falling back to deprecated behavior: obtaining relative precision from pivoted run with seed 0!")
                r = rs[0]
                srun = srun_.loc[(srun_["tags." + SEED] == seed) & (srun_["tags." + ALGORITHM + ".r"] == r)]
            assert(len(srun) == 1)
            assert_runs_deliver_precisions(srun, r)
            srun_times.append(srun["metrics." + DefaultTable.CUM_TIME].to_numpy()[0])

        plot_algo(gca, np.array(srun_times), id, performance_metric, mean_r, ts, symbol=stopped_run_symbol, color=stopped_run_color,
                  label=stopped_label, invert_axis=invert_axis)
        stopped_label = None
        if invert_axis and mean_r > lowest_r:
            gca.plot(gca.get_xlim(), mean_r * np.ones(2), diagonal_precision_style, color=diagonal_precision_color,
                     label=diag_precision_label)
            diag_precision_label = None
        id += 1

    if not invert_axis:
        gca.set_xticks(np.arange(1, id))
        gca.set_xticklabels(x_labels)
    else:
        #gca.set_yticks(np.arange(1, id))
        #gca.set_yticklabels(x_labels)
        pass
    return ts


def plot_algo(gca, values, id, metric, r, ts, symbol, color, label=None, invert_axis=False):
    def pm(a, b):
        return metric(np.sort(a), np.sort(b))

    if values.shape[0] < ts.shape[0]:
        m = np.mean(values)
        v = np.zeros(ts.shape[0])
        v[:values.shape[0]] = values
        v[values.shape[0]:] = m
        values = v
    quant = pm(ts, values)
    if not invert_axis:
        gca.plot(id * np.ones(quant.size), quant, symbol, color=color, mfc='none', label=label)
    else:
        gca.plot(quant, r * np.ones(quant.size), symbol, color=color, mfc='none', label=label)


def get_parameters_from_runs(runs):
    seeds = sorted(set(runs["tags." + SEED]))
    ds = set(runs["tags." + ALGORITHM + ".diagonal_tolerance"])
    ds.discard(None)
    ds = np.sort([float(d) for d in ds])
    return seeds, ds


def setup_plot(dataset: str, N: int, D: int, k: str, theta: float, l: float, delta: float, ds: [float], gca) -> ():
    l_disp = get_displayed_length_scale(l)
    text(1, 0.1, '\\runinfo{%s}{%i}{%i}{%s}{%f}{%0.2f}{%s}' % (latex_name(dataset), D, N, k, theta, l_disp, str(delta)),
         horizontalalignment='left', verticalalignment='bottom', fontsize='x-large')
