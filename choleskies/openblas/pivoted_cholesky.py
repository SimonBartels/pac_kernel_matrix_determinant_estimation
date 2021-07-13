import numpy as np
import warnings
from ctypes import byref, cdll, c_int, c_char, c_double

from AbstractAlgorithm import AbstractAlgorithm
from choleskies.openblas.blas_util import check_info, c_double_p, c_int_p, get_blas_object
from util.result_processing.result_tables import StoppedAlgorithmsTable, DefaultTable


def chol_pivoted_overwrite(A, tol, uplo=b'L'[0]):
    """
    pivoted Cholesky decomposition of A, in place

    NOTE:
    The upper part of A will contain the original entries.
    :param A:
        the kernel matrix plus sigma ** 2 on the diagonal
    :param tol:
        tolerance for the pivoted element
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
    piv = np.zeros(n, order='F', dtype=np.int)
    piv_p = piv.ctypes.data_as(c_int_p)
    rank = c_int(0)
    work = np.zeros(2 * n, order='F')
    work_p = work.ctypes.data_as(c_double_p)
    ret = get_blas_object().dpstrf_(byref(c_char(uplo)), byref(c_int(n)), A_p, byref(c_int(n)), piv_p, byref(rank), byref(c_double(tol)),
                       work_p, byref(info))
    info = info.value
    if info < 0:
        check_info(info)
    return ret, info, rank.value, piv, work


class PivotedCholeskyBLAS(AbstractAlgorithm):
        def get_signature(self) -> str:
            return "pcb"

        def get_latex_name(self) -> str:
            return "\\PivotedChol{}"

        def get_results_table_description(self) -> DefaultTable:
            return StoppedAlgorithmsTable()

        def init(self, diagonal_tolerance=0.) -> ():
            self.diagonal_tolerance = diagonal_tolerance

        def run_configuration(self, K_, max_iterations, max_time):
            if np.isfortran(K_):
                K = K_
            else:
                K = K_.T
            t0 = self.time()
            Cm = np.log(self.sn2)
            ret, info, rank, piv, work = chol_pivoted_overwrite(K, self.diagonal_tolerance)
            ldet = 2 * np.sum(np.log(np.diag(K[:rank, :rank])))
            U = ldet + np.sum(np.log(work[K.shape[0]+rank:]))
            L = ldet + (K.shape[0] - rank) * Cm
            self.log_metric(DefaultTable.CUM_TIME, self.time() - t0, step=rank)
            self.log_metric(DefaultTable.LOG_DET_ESTIMATE, U / 2 + L / 2, step=rank)
            self.log_metric(StoppedAlgorithmsTable.UPPER_BOUND, U, step=rank)
