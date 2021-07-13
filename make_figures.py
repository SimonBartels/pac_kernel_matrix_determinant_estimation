import mlflow
from os.path import sep
import matplotlib.pyplot as plt
from pathlib import Path

from result_management import DATASET, KERNEL
from util.visualization.visualization_utils import write_tikz_file
from visualization.speed_gain import make_legend, make_tight_joint_plot
from visualization.visualization_constants import TIKZ_OUTPUT_PATH

delta = str(0.1)
indices = [e.experiment_id for e in mlflow.tracking.MlflowClient().list_experiments()]

for i in indices:
    exp = mlflow.get_experiment(str(i))
    dataset = exp.tags[DATASET]
    kernel = exp.tags[KERNEL]
    ls = exp.tags[KERNEL + ".ls"]

    path_prefix = TIKZ_OUTPUT_PATH + sep + kernel + sep + dataset

    Path(path_prefix).mkdir(parents=True, exist_ok=True)

    plt.clf()
    fig = plt.figure()
    make_tight_joint_plot(experiment_name=exp.name, delta=delta, gca=plt.gca())
    file_name = path_prefix + sep + "joint_plot_%s.tikz" % ls
    write_tikz_file(file_name=file_name, style='resultfigstyle')

make_legend()
write_tikz_file(TIKZ_OUTPUT_PATH + sep + "legend.tikz", do_clean_figure=False, legend_name="leg:performance")
