import numpy as np

from kernel.isotropic_kernel import IsotropicKernel


def get_upper_and_lower_bounds(k: IsotropicKernel, sn2: float, X: np.ndarray):
    """
    computes an upper (Cp) and a lower bound (Cm) to log(diag(A))
    :param k:
        the kernel
    :param sn2:
        noise term, sigma ** 2
    :param X:
        dataset
    :return:
        Cp, Cm s.t. Cm <= log(diag(A)) <= Cp
    """
    Cp = np.log(k.var + sn2)
    Cm = np.log(sn2)
    return Cp, Cm
