"""
script for running all pivoted Cholesky experiments
"""
import mlflow

from choleskies.openblas.pivoted_cholesky import PivotedCholeskyBLAS
from initialize_experiments import initialize_experiment
from registry import KERNEL_DICT, ENVIRONMENT_DICT, CLIP
from result_management import get_steps_from_config, get_results_path, ENV_CPUS
from util.execution.cluster import execute_single_configuration_on_slurm_cluster, execute_job_array_on_slurm_cluster

assert(mlflow.get_tracking_uri() == get_results_path())

check_if_results_exist = False

cpus = 40

seeds = range(0, 10)
datasets = ['metro', 'tamilnadu_electricity', 'pm25', 'protein', 'bank']
ls = [-1., 0., 1., 2.]

ds = [0.005, 0.01, 0.05, 0.1, 0.5]

sn2 = 1e-3
theta = 0.

max_iterations = 60000
algo_parameters = PivotedCholeskyBLAS().get_default_parameter_dictionary()
env_dict = {ENV_CPUS: cpus}
commands = []
for seed in seeds:
    for dataset in datasets:
        for l in ls:
            for k in list(KERNEL_DICT.keys()):
                for d in ds:
                    initialize_experiment(dataset=dataset, sn2=sn2, kernel_name=k, theta=theta, l=l)
                    k_ = KERNEL_DICT[k]
                    k_.initialize(D=None, var=theta, ls=l)
                    algo_parameters["diagonal_tolerance"] = d
                    if check_if_results_exist:
                        recorded_steps = get_steps_from_config(dataset=dataset, kernel=k_, environment=env_dict,
                                                               sn2=sn2, algorithm=PivotedCholeskyBLAS(),
                                                               algo_parameters=algo_parameters,
                                                               seed=seed, clip=CLIP)
                        if len(recorded_steps) > 0:
                            assert(len(recorded_steps) == 1)
                            continue
                    command = "python run_single_configuration.py -a pcb -mi %i" % max_iterations
                    command += " -pcb-dt %f" % d
                    command += " -d %s" % dataset
                    command += " -k %s -k-ls %f -k-var %f" % (k, l, theta)
                    command += " -sn2 %f" % sn2
                    command += " -s %i" % seed
                    commands.append(command)
                    #cluster_command = execute_single_configuration_on_slurm_cluster(command=command, cpus=cpus)
                    #print("executing: %s" % cluster_command)
                    #print("with: %s" % command)
execute_job_array_on_slurm_cluster(commands, cpus=cpus)
