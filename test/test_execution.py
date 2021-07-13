import unittest
import numpy as np
import random

from util.setup_kernel_matrix import setup_kernel_matrix
from data_sets.load_dataset import load_dataset
from kernel.isotropic_kernel import OU


class TestExperimentExecution(unittest.TestCase):
    def test_computation_of_kernel_matrix_is_deterministic(self):
        seed = 0

        N = 1000
        dataset = 'metro'
        sn2 = 1e-3

        X = load_dataset(dataset)
        k = OU().initialize(X.shape[1], 0., 1.)
        # set seeds
        np.random.seed(seed)
        random.seed(seed)
        K_ = setup_kernel_matrix(k, sn2, X, N)

        # set seeds
        np.random.seed(seed)
        random.seed(seed)
        X = load_dataset(dataset)
        np.random.shuffle(X)
        X = X[:N, :]
        K = k.K(X)
        np.fill_diagonal(K, 1. + sn2)

        np.testing.assert_array_almost_equal(K, K_)

    def test_load_datasets(self):
        for ds in ['metro', 'tamilnadu_electricity', 'pm25', 'protein', 'bank']:
            load_dataset(ds)
        # seems like we can load every dataset


if __name__ == '__main__':
    unittest.main()
