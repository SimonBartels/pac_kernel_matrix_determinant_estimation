"""
script for running default Cholesky experiments
"""
import mlflow

from choleskies.openblas.default_cholesky import DefaultCholeskyBLAS
from initialize_experiments import initialize_experiment
from registry import KERNEL_DICT
from result_management import get_results_path
from util.execution.cluster import execute_single_configuration_on_slurm_cluster, execute_job_array_on_slurm_cluster
from util.execution.run_local import run_local

assert(mlflow.get_tracking_uri() == get_results_path())

cpus = 40

seeds = range(0, 10)
datasets = ['metro', 'tamilnadu_electricity', 'pm25', 'protein', 'bank']
ls = [-1., 0., 1., 2.]

sn2 = 1e-3
theta = 0.

max_iterations = 60000

template = "python run_single_configuration.py -a %s -mi %i" % (DefaultCholeskyBLAS().get_signature(), max_iterations)
commands = []
for seed in seeds:
    for l in ls:
        for k in list(KERNEL_DICT.keys()):
            for dataset in datasets:
                # make sure the experiment exists to avoid conflicts
                initialize_experiment(dataset=dataset, sn2=sn2, kernel_name=k, theta=theta, l=l)
                command = template + " -d %s" % dataset
                command += " -k %s -k-ls %f -k-var %f" % (k, l, theta)
                command += " -sn2 %f" % sn2
                command += " -s %i" % seed
                commands.append(command)
                #cluster_command = execute_single_configuration_on_slurm_cluster(command=command, cpus=cpus)
                #print("executing: %s" % cluster_command)
                #print("with: %s" % command)
execute_job_array_on_slurm_cluster(commands, cpus=cpus)
