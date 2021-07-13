"""
script for running all stochastic trace estimator experiments
"""
import gpytorch.settings as gpt_settings

from initialize_experiments import initialize_experiment
from registry import KERNEL_DICT
from stochastic_trace_estimators.gpy_torch import GPyTorch
from util.execution.cluster import execute_single_configuration_on_slurm_cluster

cpus = [40]

seeds = range(0, 10)
datasets = ['metro', 'tamilnadu_electricity', 'pm25', 'protein', 'bank', 'pumadyn']
ls = [-1., 0., 1., 2.]

sn2 = 1e-3
theta = 0.

max_iterations = gpt_settings.max_cg_iterations.value()

template = "python run_single_configuration.py -a %s -mi %i" % (GPyTorch().get_signature(), max_iterations)
for seed in seeds:
    for l in ls:
        for k in list(KERNEL_DICT.keys()):
            for dataset in datasets:
                for cpu in cpus:
                    # make sure the experiment exists to avoid conflicts
                    initialize_experiment(dataset=dataset, sn2=sn2, kernel_name=k, theta=theta, l=l)
                    command = template + " -d %s" % dataset
                    command += " -k %s -k-ls %f -k-var %f" % (k, l, theta)
                    command += " -sn2 %f" % sn2
                    command += " -s %i" % seed
                    cluster_command = execute_single_configuration_on_slurm_cluster(command=command, cpus=cpu)
                    print("executing: %s" % cluster_command)
                    print("with: %s" % command)
