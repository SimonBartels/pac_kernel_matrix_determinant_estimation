import unittest
import numpy as np
from kernel.isotropic_kernel import RBF, OU, Delta, Zero


ks = [RBF(), OU(), Delta()]  # the kernels to test


class KernelTests(unittest.TestCase):
    def test_kernel_matrix_is_fortran(self) -> ():
        """
        This test makes sure that every kernel implementation creates a Fortran aligned kernel matrix.
        :return:
            nothing
        """
        n = 10
        D = 2
        X = np.random.rand(n, D)
        for k in ks:
            K = k.initialize(D, 0., 0.).K(X)
            self.assertTrue(np.isfortran(K))

    def test_stationary_kernels(self) -> ():
        """
        Test to ensure the stationary kernel implementations have indeed properties of stationary kernels.
        :return:
            nothing
        """
        n = 10
        D = 3
        X = np.random.rand(n, D)
        for k_ in ks:
            var = np.random.randn()
            k = k_.initialize(D, var=var, ls=np.random.randn())
            K = k.K(X)
            np.testing.assert_array_almost_equal(np.diag(K), np.exp(var) * np.ones(X.shape[0]))  # make sure the matrix has a constant diagonal
            np.testing.assert_array_almost_equal(K, K.T)  # make sure the matrix is symmetric
            L = np.linalg.cholesky(K)  # make sure the matrix is positive definite

            m = int(n/2)
            K2 = k.K(X[:m, :], X[m:, :])
            np.testing.assert_array_almost_equal(K[:m, m:], K2)

    def test_RBF(self) -> ():
        """
        Test to check the distance computed for the RBF kernel.
        :return:
            nothing
        """
        X = np.arange(0, 3).T[:, None]
        K = RBF().initialize(1, 0., 0.).K(X)
        np.testing.assert_array_almost_equal(np.log(K), -np.square(X - X.T) / 2)

    def test_OU(self) -> ():
        """
        Test to check the distance computed for the OU kernel.
        :return:
            nothing
        """
        X = np.arange(0, 3).T[:, None]
        K = OU().initialize(1, 0., 0.).K(X)
        np.testing.assert_array_almost_equal(np.log(K), -np.abs(X - X.T))

    def test_neg_dist(self) -> ():
        n = 10
        D = 3
        X = np.random.rand(n, D)
        k = ks[0].initialize(D, np.random.randn(), np.random.randn())
        dist = k._get_neg_dist(X)
        m = int(n / 2)
        dist_ = k._get_neg_dist(X[:m, :], X[m:, :])
        np.testing.assert_array_almost_equal(dist[:m, m:], dist_)


if __name__ == '__main__':
    unittest.main()
