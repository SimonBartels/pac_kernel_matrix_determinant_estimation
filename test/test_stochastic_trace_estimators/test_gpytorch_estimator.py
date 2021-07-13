import unittest
import numpy as np

from data_sets.load_dataset import load_dataset
from kernel.isotropic_kernel import RBF
from util.result_processing.result_tables import DefaultTable
from util.setup_kernel_matrix import setup_kernel_matrix
from stochastic_trace_estimators.gpy_torch import GPyTorch


class TestGPySTE(unittest.TestCase):
    """
    Tests for the GPyTorch stochastic trace estimator.
    """
    def test_convergence(self):
        """
        This test makes sure that for a large preconditioner, the result of the STE will be quite correct.
        :return:
        """
        sn2 = 1e-3

        X = load_dataset("protein")
        X = X[:3000, :]  # this must be larger than 2000 or gpytorch will ignore the preconditioner...

        k = RBF()
        k.initialize(D=X.shape[1], var=0., ls=0.)

        K = setup_kernel_matrix(k, sn2, X)
        logdet = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(K))))

        gpt = GPyTorch()
        gpt.set_environment(sn2, k)
        gpt.init(num_trace_samples=10, cg_tolerance=1e-15, preconditioning_steps=500)

        def _log_metric(metric, estimate, step):
            if metric is DefaultTable.LOG_DET_ESTIMATE:
                rel_err = (estimate - logdet) / logdet
                self.assertLess(np.abs(rel_err), 1e-2)

        gpt.log_metric = _log_metric
        gpt.run_configuration(K, max_iterations=X.shape[0], max_time=None)


if __name__ == '__main__':
    unittest.main()
