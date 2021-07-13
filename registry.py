import os
import subprocess


VERBOSE = False
CLIP = 0.

ENVIRONMENT_DICT = dict()
KERNEL_DICT = dict()
ALGORITHM_DICT = dict()


def populate_registry():
    from result_management import ENV_CPUS, ENV_PROC

    """
    Add algorithms to registry.
    Depending on which packages are installed or not, we might skip a few.
    """
    from choleskies.openblas.default_cholesky import DefaultCholeskyBLAS
    from choleskies.openblas.pivoted_cholesky import PivotedCholeskyBLAS
    from choleskies.openblas.stopped_cholesky import StoppedCholeskyBLAS
    _algorithm_list = [DefaultCholeskyBLAS(), StoppedCholeskyBLAS(), PivotedCholeskyBLAS()]

    try:
        from stochastic_trace_estimators.gpy_torch import GPyTorch
        _algorithm_list.append(GPyTorch())
    except ImportError as e:
        pass

    for a in _algorithm_list:
        ALGORITHM_DICT[a.get_signature()] = a
    assert(len(ALGORITHM_DICT.keys()) == len(_algorithm_list))  # make sure we didn't by accident use the same signature twice

    from kernel.isotropic_kernel import RBF, OU
    KERNEL_DICT[RBF().name] = RBF()
    KERNEL_DICT[OU().name] = OU()

    assert(len(KERNEL_DICT.keys()) == 2)


    ENVIRONMENT_DICT[ENV_CPUS] = len(os.sched_getaffinity(os.getpid()))  # count the number of CPUs
    ENVIRONMENT_DICT[ENV_PROC] = (subprocess.check_output("lscpu | grep 'Model name'", shell=True).strip()).decode()  #platform.processor()


populate_registry()
