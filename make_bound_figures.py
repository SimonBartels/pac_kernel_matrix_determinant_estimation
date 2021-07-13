import numpy as np
from os.path import sep
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path

from result_management import DATASET, get_results_path, KERNEL
from util.visualization.visualization_utils import write_tikz_file
from visualization.precision_over_steps import make_plot, make_bound_legend
from visualization.visualization_constants import TIKZ_OUTPUT_PATH
from visualization.speed_gain import make_inverse_plot


delta = str(0.1)
#indices = [int(e.experiment_id) for e in mlflow.tracking.MlflowClient().list_experiments()]
#ndices = np.array([6, 24])  # creates only the figures present in the paper

for i in indices:
    exp = mlflow.get_experiment(str(i))
    dataset = exp.tags[DATASET]
    kernel = exp.tags[KERNEL]
    ls = exp.tags[KERNEL + ".ls"]

    path_prefix = TIKZ_OUTPUT_PATH + sep + kernel + sep + dataset

    Path(path_prefix).mkdir(parents=True, exist_ok=True)

    make_inverse_plot(experiment_name=exp.name, delta=delta, gca=plt.gca())
    write_tikz_file(path_prefix + sep + "inverse_plot_%s.tikz" % ls)
    #plt.show()
    plt.clf()

    make_plot(experiment_name=exp.name, delta=delta)
    plt.legend()
    write_tikz_file(path_prefix + sep + "bound_%s.tikz" % ls)
    #plt.show()
    plt.clf()

make_bound_legend()
write_tikz_file(TIKZ_OUTPUT_PATH + sep + "bound_legend.tikz", do_clean_figure=False, legend_name="leg:bound")
