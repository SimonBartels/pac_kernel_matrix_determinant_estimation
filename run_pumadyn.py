"""
script to run all pumadyn configurations
"""
import gpytorch.settings as gpt_settings

import registry
from choleskies.openblas.default_cholesky import DefaultCholeskyBLAS
from choleskies.openblas.pivoted_cholesky import PivotedCholeskyBLAS
from choleskies.openblas.stopped_cholesky import StoppedCholeskyBLAS
from registry import KERNEL_DICT
from result_management import ENV_CPUS
from stochastic_trace_estimators.gpy_torch import GPyTorch
from util.execution.run_local import run_local
from util.result_processing.diagonal_tolererance_to_precision import get_precisions_from_diagonal_tolerance

dataset = 'pumadyn'
max_iterations = 10000
sn2 = 1e-3
theta = 0.
seeds = [i for i in range(0, 10)]
ds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
ls = [-1., 0., 1., 2.]
ks = list(KERNEL_DICT.keys())


max_cg_iterations = gpt_settings.max_cg_iterations.value()
cpus = registry.ENVIRONMENT_DICT[ENV_CPUS]
block_size = 256 * cpus

alg_list = [(DefaultCholeskyBLAS(), ""), (GPyTorch(), "-mi %i" % max_cg_iterations)]
for d in ds:
    alg_list.append((PivotedCholeskyBLAS(), "-pcb-dt %f" % d))

command_template = "python run_single_configuration.py -a %s %s -d %s -k %s -k-ls %s -s %i"
for l in ls:
    for k in ks:
        for alg, alg_params in alg_list:
            for s in seeds:
                command = command_template % (alg.get_signature(), alg_params, dataset, k, str(l), s)
                run_local(command)
        for dt in ds:
            kfunc = KERNEL_DICT[k]
            kfunc.initialize(D=None, var=theta, ls=l)
            stopped_params_ = "-scb-bs %i -scb-ib %i" % (block_size, block_size)
            for s in seeds:
                rs = get_precisions_from_diagonal_tolerance(dataset=dataset, kfunc=kfunc, sn2=sn2, seed=s, dt=dt,
                                                            env=registry.ENVIRONMENT_DICT)
                assert (len(rs) == 1)
                stopped_params = stopped_params_ + " -scb-r %f" % rs[0]
                command = command_template % (StoppedCholeskyBLAS().get_signature(), stopped_params, dataset, k, str(l), s)
                run_local(command)
