import unittest
import random
import numpy as np

import result_management
result_management.save_diagonal = lambda *args: None

from choleskies.openblas.default_cholesky import DefaultCholeskyBLAS
from choleskies.openblas.pivoted_cholesky import PivotedCholeskyBLAS
from choleskies.openblas.stopped_cholesky import StoppedCholeskyBLAS, chol_stopped_overwrite
from data_sets.load_dataset import load_dataset
from kernel.isotropic_kernel import RBF
from util.result_processing.result_tables import DefaultTable
from util.stop_cond import inv_HN


class TestCholeskyImplementations(unittest.TestCase):
    def test_choleskies(self):
        """
        Tests for all Choleskies in our repertoire that the desired precision is achieved on random datasets.
        In theory, this could fail for our stopped Cholesky but it shouldn't in practice.
        """
        chols = [StoppedCholeskyBLAS(), PivotedCholeskyBLAS(), DefaultCholeskyBLAS()]
        n = 511
        sn2 = 1e-6
        K = np.random.rand(n, n)
        K = K.dot(K.T) + sn2 * np.eye(n)  # make matrix s.p.d.
        log_det = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(K))))

        def log_check(parameter, value, step):
            if step == n and parameter == DefaultTable.LOG_DET_ESTIMATE:
                self.assertAlmostEqual(value, log_det, places=6)

        for chol in chols:
            chol.log_metric = log_check
            k = RBF().initialize(1, var=np.log(np.max(np.diag(K))), ls=0.)
            chol.set_environment(sn2=sn2, k=k)
            chol.init()
            L = K.copy()
            chol.run_configuration(L, n, np.infty)

    def test_stopped_cholesky_when_not_stopping(self):
        """
        When not stopping we should obtain the exact Cholesky.
        """
        n = 511
        sn2 = 1e-3
        K = np.random.rand(n, n)
        K = K.dot(K.T) + sn2 * np.eye(n)  # make matrix s.p.d.
        chols = [StoppedCholeskyBLAS()]

        for chol in chols:
            chol.log_metric = lambda *args, **kwargs: None
            k = RBF().initialize(1, var=np.log(np.max(np.diag(K))), ls=0.)
            chol.set_environment(sn2=sn2, k=k)
            chol.init(r=0.)
            L = K.copy().T  # make fortran contiguous
            chol.run_configuration(L, n, max_time=np.inf)
            L = np.tril(L)
            np.testing.assert_array_almost_equal(K, L.dot(L.T))

    def test_stopped_cholesky_return_values(self):
        """
        Tests the BLAS wrapper for the stopped Cholesky.
        """
        n = 7
        f = 1e-6
        r = 0  # make sure early stopping is not possible
        delta = 0.1
        K = f * np.eye(n)
        Cm = np.log(f)
        Cp = np.log(1 + f)
        C = Cp - Cm
        ret, info, est, m, mean = chol_stopped_overwrite(A=K, r=r, lnSmallestEval=Cm, C=C,
                                                         c_hinv=C * inv_HN(delta, K.shape[0]))
        self.assertTrue(info == 0)
        self.assertEqual(m, n)
        self.assertAlmostEqual(est, n * np.log(f))
        self.assertAlmostEqual(mean, np.log(f))


if __name__ == '__main__':
    unittest.main()
