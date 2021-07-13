import numpy as np
import warnings
from ctypes import byref, c_int, c_char

from AbstractAlgorithm import AbstractAlgorithm
from choleskies.openblas.blas_util import check_info, get_blas_object, c_double_p
from result_management import save_diagonal
from util.result_processing.result_tables import DefaultCholeskyTable, DefaultTable


def chol_default_overwrite(A, uplo=b'L'[0]):
    """
    default Cholesky decomposition of A, in place

    NOTE:
    The upper part of A will contain the original entries.
    :param A:
        the kernel matrix plus sigma ** 2 on the diagonal
    :param uplo:
        byte character of either 'L' or 'U', whether to compute upper or lower diagonal
    :return:
        return value of the call, debug information from BLAS
    """
    if not np.isfortran(A):
        warnings.warn('Matrix is not FORTRAN contiguous!', RuntimeWarning)
    n = A.shape[0]
    A_p = A.ctypes.data_as(c_double_p)
    info = c_int(0)
    ret = get_blas_object().dpotrf_(byref(c_char(uplo)), byref(c_int(n)), A_p, byref(c_int(n)), byref(info))
    info = info.value
    check_info(info)
    return ret, info


class DefaultCholeskyBLAS(AbstractAlgorithm):
    def get_signature(self) -> str:
        return "dcb"

    def init(self) -> ():
        pass

    def get_results_table_description(self) -> DefaultTable:
        return DefaultCholeskyTable()

    def run_configuration(self, K_, max_iterations, max_time):
        if np.isfortran(K_):
            K = K_
        else:
            K = K_.T
        t0 = self.time()
        chol_default_overwrite(K)
        ldet = 2 * np.sum(np.log(np.diag(K)))
        if np.isnan(ldet):
            # somehow the run crashed -- let's not record the result
            return
        self.log(step=K.shape[0], estimate=ldet, t0=t0)
        save_diagonal(K)
