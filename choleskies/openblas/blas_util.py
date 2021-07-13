import os
import warnings
import ctypes
from ctypes import cdll, c_int, c_double


if not 'blas' in vars():
    blas_path = os.path.join(os.path.join(os.path.dirname(__file__), 'lib'), 'libopenblas.so')  # path to library
    ##num_cpus = mlflow.get_run(mlflow.active_run().info.run_id).data.tags[ENV_CPUS]
    #num_cpus = ENVIRONMENT_DICT[ENV_CPUS]
    #blas_path = os.path.join(os.path.dirname(__file__), 'libopenblas_threads_%i.so' % num_cpus)  # path to library
    blas = cdll.LoadLibrary(blas_path)  # load library with ctypes
    c_double_p = ctypes.POINTER(c_double)  # convenience definition of a pointer to a double
    c_int_p = ctypes.POINTER(c_int)


def get_blas_object():
    return blas


def check_info(info: int) -> ():
    """
    takes debug information from the BLAS Cholesky decomposition and prints its meaning
    :param info:
        the debug info
    :return:
        nothing
    """
    if info != 0:
        if info < 0:
            warnings.warn("Function call parameters are misspecified. Info value: %i" % info, RuntimeWarning)
        else:
            warnings.warn("K stopped being s.p.d. with row %i." % info, RuntimeWarning)