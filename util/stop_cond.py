import numpy as np
from scipy.optimize import fminbound


def compute_bounds(N: int, run_length: int, diagL: np.ndarray, Cm: float, Cp: float, delta: float) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Computes the bounds and the stopping conditions for a given run.
    This function is used for testing and visualization.
    :param N:
        size of the dataset
    :param run_length:
        number of iterations the Cholesky ran (before crashing)
    :param diagL:
        diagonal of the Cholesky decomposition
    :param Cm:
        lower bound on log(diag(A))
    :param Cp:
        upper bound on log(diag(A))
    :param delta:
        desired failure chance
    :return:
        means, lower bounds, deterministic upper bounds, deterministic stopping conditions, probabilistic upper bounds,
        probabilistic stopping conditions
        each as numpy array of length run_length
    """
    C = Cp - Cm  # convenience constant
    Hinv = inv_HN(delta / 2, N)  # constant for error tolerance
    f_seq = 2 * np.log(diagL[:run_length])  # sequence of elements f
    log_dets = np.cumsum(f_seq)  # sequence of log determinants
    stop_cond = np.nan * np.ones([run_length, ])  # stopping condition
    stop_cond_det = np.nan * np.ones([run_length, 1])  # deterministic stopping condition
    UBd = np.zeros([run_length, 1])  # deterministic upper bound
    LBd = np.zeros([run_length, 1])  # deterministic lower bound
    UB = np.zeros([run_length, 1])  # probabilistic upper bound
    mus = np.zeros([run_length, 1])  # mean estimates

    for t in range(0, run_length):
        LBd[t] = log_dets[t] + (N - t - 1) * Cm
        UBd[t] = log_dets[t] + (N - t - 1) * Cp
        stop_cond_det[t] = (UBd[t] - LBd[t]) / 2 / min(abs(UBd[t]), abs(LBd[t]))

        mu = np.mean(f_seq[:t+1])
        mus[t] = mu
        e_j = C * Hinv * (1 + (N - t - 1) / (t + 1))  # error tolerance in step t (+ 1)
        UB[t] = min(LBd[t] + (N - t - 1) * (mu - Cm) + e_j, UBd[t])
        stop_cond[t] = (UB[t] - LBd[t]) / 2 / min(abs(UB[t]), abs(LBd[t]))

    return mus, LBd, UBd, stop_cond_det, UB, stop_cond


def HN(x: float, n: int) -> float:
    """
    function $H_N(x)$ from the paper
    :param x:
        excess
    :param n:
        length of the sequence
    :return:
        log of the probability that the sum exceeds x
    """
    return H(x, n, n)


def H(x: float, v2: float, n: int) -> float:
    """
    function $H_n(x, v)$ from Fan et al.'s Hoeffding bound
    :param x:
        excess
    :param v2:
        bound on the conditional variance
    :param n:
        length of the sequence
    :return:
        log of the probability that the sum exceeds x while the conditional variance is bounded by v2
    """
    logP = (n / (n + v2)) * ((x + v2) * np.log(v2 / (x + v2)) + (n - x) * np.log(n / (n - x)))
    return logP


def inv_HN(delta: float, N: int) -> float:
    """
    Returns the excess, that the sum can pass with probability less than delta.
    :param delta:
        the desired failure tolerance
    :param N:
        length of the sequence
    :return:
        the excess x, s.t. exp(HN(x, n)) <= delta
    """
    if delta >= 1.:
        return 0.

    # just optimize the difference of exp(HN(x, N)) and delta
    Hinv = fminbound(lambda x: (HN(x, N) - np.log(delta)) ** 2, 0, N)

    # if we did not find a reasonably close x ...
    if np.exp(HN(Hinv, N)) > delta + 1e-6:
        # ... we have to set x to N
        Hinv = N
    return Hinv


def inv_H(delta: float, v2: float, N: int) -> float:
    """
    Returns the excess, that the sum can pass with probability less than delta.
    :param delta:
        the desired failure tolerance
    :param v2:
        bound on the variance
    :param N:
        length of the sequence
    :return:
        the excess x, s.t. exp(HN(x, n)) <= delta
    """
    if delta >= 1.:
        return 0.

    # just optimize the difference of exp(HN(x, N)) and delta
    Hinv = fminbound(lambda x: (H(x, v2, N) - np.log(delta)) ** 2, 0, N)

    # if we did not find a reasonably close x ...
    if np.exp(H(Hinv, v2, N)) > delta + 1e-6:
        # ... we have to set x to N
        Hinv = N
    return Hinv