"""
Script to create the tikz figure for the stochastic trace estimator results.
"""
import mlflow
import matplotlib.pyplot as plt
from os.path import sep

from choleskies.openblas.default_cholesky import DefaultCholeskyBLAS
from registry import KERNEL_DICT
from stochastic_trace_estimators.gpy_torch import GPyTorch
from result_management import ALGORITHM, KERNEL, DATASET, SEED, find_experiments_with_tags, \
    experiment_name_is_consistent
from util.result_processing.result_tables import DefaultTable
from data_sets.load_dataset import latex_name
from util.visualization.visualization_utils import write_tikz_file
from visualization.visualization_constants import *
from tikzplotlib_fix.tikzplotlib_fix import fix_tikzplotlib_clean_figure
from visualization.visualization_constants import TIKZ_OUTPUT_PATH

fix_tikzplotlib_clean_figure()  # apply patch to fix some bugs in tikzplotlib.clean_figure()


def plot_precisions_for_indices(indices):
    ax3 = plt.subplot(111)
    x_labels = []
    id = 0
    for i in indices:
        experiment = mlflow.get_experiment(str(i))
        mlflow.set_experiment(experiment_name=experiment.name)
        runs = mlflow.search_runs()
        # runs with default Cholesky
        def_runs = runs.loc[(runs["tags." + ALGORITHM] == DefaultCholeskyBLAS().get_signature())]
        ground_truth = def_runs["metrics." + DefaultTable.LOG_DET_ESTIMATE].to_numpy()
        assert(np.std(ground_truth) < 1e-7)
        ground_truth = np.mean(ground_truth)

        k = experiment.tags[KERNEL]
        ls = float(experiment.tags[KERNEL + ".ls"])
        dataset = experiment.tags[DATASET]

        id = id + 1
        x_labels.append("\\expconfig{%s}{%s}{%s}" % (latex_name(dataset), k, ls))
        # add stochastic trace estimator
        sruns = runs.loc[(runs["tags." + ALGORITHM] == GPyTorch().get_signature())]
        values = np.abs((sruns["metrics." + DefaultTable.LOG_DET_ESTIMATE].to_numpy() - ground_truth) / ground_truth)
        #ax3.boxplot(values, positions=[id])
        ax3.plot(id * np.ones(len(values)), values, 'x', color=gra)
    ax3.set_xticks(np.arange(1, len(x_labels)+1))
    ax3.set_xticklabels(x_labels)
    ax3.set_ylim([1e-3, 1])
    #ax3.set_yticks([10**-i for i in range(1, 4)])
    ax3.set_yscale("log")
    # Unfortunately, matplotlib2tikz does not know about axes transforms.
    #ax3.text(0.01, 0.01, latex_name(dataset), transform=ax3.transAxes, horizontalalignment='left', verticalalignment='bottom')
    ax3.text(1, 5*1e-3, latex_name(dataset), horizontalalignment='left', verticalalignment='bottom')
    ax3.set_xlabel("\\labelx{}")
    ax3.set_ylabel("\\labely{}")

kernel = 'RBF'
datasets = ['metro', 'tamilnadu_electricity', 'pm25', 'protein', 'bank', 'pumadyn']
ls = [-1., 0., 1., 2.]
for d in datasets:
    indices = []
    for l in ls:
        k = KERNEL_DICT[kernel].initialize(None, var=0., ls=float(l))
        experiments = find_experiments_with_tags(d, k, environment={}, sn2=1e-3, clip=0.0)
        assert(len(experiments) > 0)
        runs = mlflow.search_runs([e.experiment_id for e in experiments])
        ids = np.array([int(i) for i in runs['experiment_id'].to_numpy()
                        if experiment_name_is_consistent(mlflow.get_experiment(i))])
        assert(np.all(ids == ids[0]))
        indices.append(ids[0])
    fig = plt.figure()
    plot_precisions_for_indices(indices)
    file_name = TIKZ_OUTPUT_PATH + sep + "ste_%s_%s.tikz" % (d, kernel)
    write_tikz_file(file_name=file_name, gca=plt.gca())
    fig.clf()
