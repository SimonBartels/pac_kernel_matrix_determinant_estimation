import numpy as np

from kernel.isotropic_kernel import IsotropicKernel


def swap_rows(A, i, j):
    temp_r_i = A[i, :].copy()
    A[i, :] = A[j, :]
    A[j, :] = temp_r_i


def ref_pivoted_cholesky(k: IsotropicKernel, X: np.ndarray, sn2: float, max_iterations: int):
    C = np.zeros([X.shape[0], max_iterations])
    log_dets = np.zeros(max_iterations)
    Us = np.zeros(max_iterations)

    d = k.Kdiag(X) + sn2
    p = np.arange(0, X.shape[0])


    # since we only consider stationary kernels, the first chosen index is 0
    C[0, 0] = np.sqrt(d[0])
    C[1:, :1] = k.K(X[1:, :], X[:1, :])
    C[1:, 0] /= C[0, 0]

    d[1:] -= np.square(C[1:, 0])

    det = 2 * np.log(C[0, 0])
    log_dets[0] = det
    Us[0] = det + np.sum(np.log(d[1:]))

    for m in range(1, max_iterations):
        i = m + np.argmax(d[m:])
        if i != m:
            swap_rows(C, m, i)
            t = p[i]
            p[i] = p[m]
            p[m] = t

        C[m, m] = np.sqrt(d[i])
        d[i] = d[m]

        C[m+1:, [m]] = k.K(X[p[m+1:], :], X[[p[m]], :])
        C[m+1:, m] -= np.sum(C[m+1:, :m].dot(np.expand_dims(C[m, :m], -1)), axis=1)
        C[m+1:, m] /= C[m, m]

        d[m+1:] -= np.square(C[m+1:, m])

        det += 2 * np.log(C[m, m])
        log_dets[m] = det
        Us[m] = det + np.sum(np.log(d[m + 1:]))

    return C, log_dets, Us, p, d
