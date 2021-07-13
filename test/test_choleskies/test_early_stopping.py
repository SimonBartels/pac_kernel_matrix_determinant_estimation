import unittest
import random
import numpy as np

from choleskies.openblas.stopped_cholesky import chol_stopped_overwrite, StoppedCholeskyBLAS
from choleskies.ref.stopped_chol import ref_stopped_cholesky
from data_sets.load_dataset import load_dataset
from kernel.isotropic_kernel import RBF
from util.result_processing.result_tables import DefaultTable, StoppedAlgorithmsTable
from util.stop_cond import compute_bounds, inv_HN


class TestStoppedCholeskies(unittest.TestCase):
    def test_blas_impl_on_pumadyn(self):
        """
        Tests the BLAS implementation against the reference implementation on PUMADYN.
        """
        r = 0.727886
        delta = 0.1
        block_size = 2048

        seed = 0
        np.random.seed(seed)
        random.seed(seed)

        X = load_dataset("pumadyn")
        sn2 = 1e-3
        k = RBF().initialize(D=X.shape[1], var=0.0, ls=2.0)
        K = k.K(X) + sn2 * np.eye(X.shape[0])

        diagL = np.diag(np.linalg.cholesky(K))
        mus, LBd, UBd, stop_cond_det, UB, stop_cond = compute_bounds(N=X.shape[0], run_length=X.shape[0],
                                                                     diagL=diagL,
                                                                     Cm=np.log(sn2), Cp=np.log(sn2 + k.var),
                                                                     delta=delta)
        estimates = LBd / 2 + UB / 2
        stopping_point = np.argmax(stop_cond < r)
        stopping_point = int(np.ceil(stopping_point / block_size) * block_size)

        last_stop = np.zeros(1, dtype=np.int)

        def log_check(parameter, value, step):
            last_stop[0] = step
            if parameter == DefaultTable.LOG_DET_ESTIMATE:
                self.assertAlmostEqual(value, estimates[step-1, 0], places=5)
            if parameter == StoppedAlgorithmsTable.UPPER_BOUND:
                self.assertAlmostEqual(value, UB[step-1, 0], places=5)

        chol = StoppedCholeskyBLAS()
        chol.log_metric = log_check
        chol.set_environment(sn2=sn2, k=k)
        chol.init(r=r, delta=delta, block_size=block_size, initial_block=block_size)
        chol.run_configuration(K, np.infty, np.infty)
        self.assertEqual(stopping_point, last_stop[0])

    def test_zeros_dataset(self) -> ():
        """
        Another test for a scenario where the stopped Cholesky should indeed stop.
        """
        n = 2 * 1025
        D = 2
        r = 0.01
        delta = 0.1
        sn2 = 1e-3
        X = np.zeros((n, D))
        K = RBF().initialize(D, 0., 0.).K(X)
        np.fill_diagonal(K, 1. + sn2)
        Cm = np.log(sn2)
        Cp = np.log(1 + sn2)
        C = Cp - Cm
        ret, info, est, m, mean = chol_stopped_overwrite(A=K, r=r, lnSmallestEval=Cm, C=C, c_hinv=C * inv_HN(delta/2, n),
                                                         block_size=512, initial_block=512)
        self.assertLess(m, n)

    def test_zeros_dataset_no_stopping(self) -> ():
        """
        A test for a scenario where the stopped Cholesky MUST NOT stop.
        """
        n = 2 * 1025
        D = 2
        r = 0.0  # when r=0 stopping should be impossible
        delta = 0.1
        sn2 = 1e-3
        X = np.zeros((n, D))
        K = RBF().initialize(D, 0., 0.).K(X)
        np.fill_diagonal(K, 1. + sn2)
        Cm = np.log(sn2)
        Cp = np.log(1 + sn2)
        C = Cp - Cm
        ret, info, est, m, mean = chol_stopped_overwrite(A=K, r=r, lnSmallestEval=Cm, C=C, c_hinv=C * inv_HN(delta/2, n),
                                                         block_size=512, initial_block=512)
        self.assertEqual(m, n)

    def test_against_reference_implementation(self):
        """
        Tests the BLAS implementation against the reference implementation on randomly generated datasets.
        """
        n = 4096
        block_size = 2048
        D = 2
        r = 0.1
        delta = 0.25
        sn2 = 1e-3
        X = np.random.rand(n, D)
        K = RBF().initialize(D, 0., 1.).K(X)
        np.fill_diagonal(K, 1. + sn2)
        Cm = np.log(sn2)
        Cp = np.log(1 + sn2)
        C = Cp - Cm
        c_hinv = C * inv_HN(delta / 2, K.shape[0])
        ret, info, est, m, mean = chol_stopped_overwrite(A=K, r=r, lnSmallestEval=Cm, C=C, c_hinv=c_hinv,
                                                         block_size=block_size,
                                                         initial_block=block_size)
        diagL = np.diag(K)
        mus, LBd, UBd, stop_cond_det, UB, stop_cond_r = compute_bounds(n, n, diagL, Cm, Cp, delta)
        self.assertLess(m, n)
        np.testing.assert_almost_equal(mean, mus[m-1])
        np.testing.assert_almost_equal(est, 0.5 * LBd[m-1] + 0.5 * UB[m-1])

    def test_reference_implementation(self) -> ():
        """
        Tests the reference implementation of the stopped Cholesky.
        For a dataset of all zeros, early stopping should occur and the estimate should be correct.
        :return:
            nothing
        """
        n = 2 * 1025
        D = 2
        r = 0.001
        delta = 0.1
        sn2 = 1e-3
        X = np.zeros((n, D))
        k = RBF().initialize(D, 0., 0.)
        K = k.K(X) + sn2 * np.eye(n)
        diagL = np.diag(np.linalg.cholesky(K))
        true_log_det = 2 * np.sum(np.log(diagL))
        Cm = np.log(sn2)
        Cp = np.log(np.max(np.diag(K)))
        Hinv = (Cp - Cm) * inv_HN(delta / 2, n)
        est, U, L, l, m, mean = ref_stopped_cholesky(K, r, n, Cm, Cp, Hinv)
        self.assertLess(m, n)
        self.assertLess(np.abs((est - true_log_det) / true_log_det), r)

        mus, LBd, UBd, stop_cond_det, UB, stop_cond_r = compute_bounds(n, n, diagL, Cm, Cp, delta)
        np.testing.assert_allclose(mus[m-1], mean, rtol=1e-5)
        np.testing.assert_allclose(LBd[m-1], L, rtol=1e-5)
        np.testing.assert_allclose(UB[m-1], U, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
