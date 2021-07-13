import numpy as np


def ref_stopped_cholesky(K, r, max_iterations, Cm, Cp, Hinv):
    N = K.shape[0]
    K = np.linalg.cholesky(K)

    m = 0
    det = 0
    U = N * Cp
    L = N * Cm
    for m in range(1, max_iterations+1):
        det += 2 * np.log(K[m-1, m-1])
        U = det + min((N - m) * (det + Hinv) / m + Hinv, (N - m) * Cp)
        L = det + (N - m) * Cm
        if np.sign(U) == np.sign(L) and U - L < 2 * r * min(np.abs(U), np.abs(L)):
            break

    estimate = U / 2 + L / 2

    assert(U >= L)

    l = np.diag(K[:m-1, :m-1])
    mean = np.mean(2 * np.log(l)).flatten()

    return estimate, U, L, l, m, mean
