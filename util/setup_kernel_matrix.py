import numpy as np
from kernel.isotropic_kernel import IsotropicKernel


def setup_kernel_matrix(kernel: IsotropicKernel, sn2: float, x: np.ndarray, clip=0.) -> np.ndarray:
    """
    computes the kernel matrix after shuffling the dataset
    :param kernel:
        the kernel
    :param sn2:
        noise term to be added to the diagonal
    :param x:
        the dataset
    :param clip:
        minimal value for kernel matrix entries
    :return:
        the kernel matrix
    """
    np.random.shuffle(x)
    K = kernel.K(x)
    K[np.diag_indices_from(K)] += sn2
    np.clip(K, clip, np.infty, out=K)
    return K
