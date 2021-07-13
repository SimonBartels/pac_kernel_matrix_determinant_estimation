import numpy as np
import warnings
from ctypes import byref, c_int, c_char, c_double, c_long

from AbstractAlgorithm import AbstractAlgorithm
from choleskies.openblas.blas_util import check_info, c_double_p, get_blas_object
from util.result_processing.result_tables import StoppedAlgorithmsTable, DefaultTable
from util.stop_cond import inv_HN


def chol_stopped_overwrite(A, r, lnSmallestEval, C, c_hinv, block_size=8192, initial_block=8192, impl=b'B'[0],
                           n=None, m=None) -> (int, int, float, int, float):
    """
    stopped Cholesky decomposition of A, in place

    NOTE:
    The upper part of A will contain the original entries.
    If the Cholesky decomposition stops in time tau, this Cholesky is contained in A[:tau, :tau].
    :param A:
        the kernel matrix plus sigma ** 2 on the diagonal
    :param r:
        the desired precision
    :param lnSmallestEval:
        log of the smallest Eigenvalue of A, i.e. log(sn2)
    :param C:
        difference between upper and lower bound on log(diag(A))
    :param c_hinv:
        C * H^-1(delta/2)
    :param impl:
        byte character of either 'A' or 'B', whether to use the default implementation (A) or Banachiewicz (B)
    :param n:
        number of columns of A
    :param m:
        number of rows of A
    :return:
        return value of the call, debug information from BLAS, log-determinant estimate, size of the subset used,
        last mean estimate computed
    """
    if impl != b'B'[0] and impl != b'A'[0]:
        raise ValueError("The implementation flag must be either %i for the default implementation or %i for the "
                         "Banachiewicz implementation. Got instead: %i" % (b'A'[0], b'B'[0], impl))
    if not np.isfortran(A):
        warnings.warn('Matrix is not FORTRAN contiguous!', RuntimeWarning)
    A_p = A.ctypes.data_as(c_double_p)  # pointer to A
    estimate = np.zeros([1, 1], order='F')  # this array will contain the estimate for the log-determinant
    estimate_p = estimate.ctypes.data_as(c_double_p)  # pointer to above array
    mean = np.zeros([1, 1], order='F')  # this array will contain the last mean estimate computed
    mean_p = mean.ctypes.data_as(c_double_p)  # pointer to above array
    info = c_int(0)  # integer that will contain debug information
    subset_size = c_int(0)  # integer that will contain the stopping point

    # the n and m parameters are for debugging, they describe the shape of A
    if n is None:
        n = A.shape[0]
    if m is None:
        m = n

    ret = get_blas_object().dpotrfp_(byref(c_char(impl)), byref(c_int(n)), A_p, byref(c_int(m)),
                                     byref(info), c_double(r), c_double(c_hinv),
                                     c_double(lnSmallestEval), c_double(C), c_long(block_size), c_long(initial_block),
                                     estimate_p,
                                     byref(subset_size), mean_p)  # call to the Cholesky decomposition
    check_info(info.value)  # make sure everything went smoothly
    return ret, info.value, estimate[0, 0], subset_size.value, mean[0, 0]


class StoppedCholeskyBLAS(AbstractAlgorithm):
    def get_signature(self) -> str:
        return "scb"

    def get_latex_name(self) -> str:
        return "\\StoppedChol{}"

    def get_results_table_description(self) -> DefaultTable:
        return StoppedAlgorithmsTable()

    def init(self, r=0., delta=0.1, block_size=8192, initial_block=8192) -> ():
        self.r = r
        self.delta = delta
        self.block_size = block_size
        self.initial_block = initial_block
        self._blas_call = chol_stopped_overwrite

    def run_configuration(self, K_, max_iterations, max_time):
        if np.isfortran(K_):
            K = K_
        else:
            K = K_.T
        t0 = self.time()
        Cm = np.log(self.sn2)
        Cp = np.log(self.k.var + self.sn2)
        C = Cp - Cm
        c_hinv = C * inv_HN(self.delta / 2, K.shape[0])  # error constant from the paper
        initial_block = self.initial_block
        if initial_block > K.shape[0]:
            initial_block = K.shape[0]
        _, info, estimate, step, mean = self._blas_call(K, self.r, Cm, C, c_hinv, self.block_size, initial_block)
        self.log_metric(DefaultTable.CUM_TIME, self.time() - t0, step=step)
        self.log_metric(DefaultTable.LOG_DET_ESTIMATE, estimate, step=step)
        """
        The following computations have already been executed but we do not return them, that's why we ignore them here.
        They are peanuts anyway.
        """
        ldet = 2 * np.sum(np.log(np.diag(K[:step, :step])))
        U = ldet + (K.shape[0] - step) * (ldet + c_hinv) / step + c_hinv
        U = min(U, ldet + (K.shape[0] - step) * Cp)
        self.log_metric(StoppedAlgorithmsTable.UPPER_BOUND, U, step=step)
