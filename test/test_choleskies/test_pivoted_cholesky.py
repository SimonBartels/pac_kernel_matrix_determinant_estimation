import unittest
import numpy as np

from choleskies.openblas.pivoted_cholesky import chol_pivoted_overwrite
from choleskies.ref.pivoted_chol import ref_pivoted_cholesky
from kernel.isotropic_kernel import RBF


class TestPivotedCholesky(unittest.TestCase):
    def test_equivalence_to_lapack_impl(self):
        D = 3
        n = 10
        X = np.random.randn(n, D)
        k = RBF().initialize(D, np.random.randn(), np.random.randn())
        sn2 = 1e-3

        L = k.K(X)
        np.fill_diagonal(L, L[0, 0] + sn2)
        chol_pivoted_overwrite(L, tol=0)
        L = np.tril(L)

        #m = int(n/2)
        m = n
        L_, _, _, p = ref_pivoted_cholesky(k, X, sn2, m)
        print(L)
        print(L_)
        np.testing.assert_array_almost_equal(L[:, :m], L_)



if __name__ == '__main__':
    unittest.main()
