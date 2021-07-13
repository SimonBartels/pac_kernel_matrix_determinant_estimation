import numpy as np

from kernel.fdot import fdot


class IsotropicKernel:
    def __init__(self):
        self.name = type(self).__name__

    def initialize(self, D, var, ls):
        self.D = D
        self.var = np.exp(var)  # amplitude
        self.ls = np.exp(2 * ls)  # we store the squared length-scale
        self.parameters = {"var": var, "ls": ls}
        return self

    def K(self, X, X2=None):
        K = self._K(self._get_neg_dist(X, X2))
        K *= self.var
        return K

    def Kdiag(self, X):
        return self.var * np.ones(X.shape[0])

    def _K(self, dist):
        raise NotImplementedError("abstract method")

    def _get_neg_dist(self, X, X2=None):
        Xsq = np.sum(np.square(X), 1)
        if X2 is None:
            K = 2 * fdot(X, X)  # X * X.T
            K -= Xsq[:, None]
            K -= Xsq[None, :]
            np.fill_diagonal(K, 0.)
        else:
            X2sq = np.sum(np.square(X2), 1)
            K = 2 * fdot(X, X2) - (Xsq[:, None] + X2sq[None, :])
        K /= self.ls
        return K


class RBF(IsotropicKernel):
    def _K(self, K):
        K /= 2.
        np.exp(K, out=K)
        return K


class OU(IsotropicKernel):
    def _K(self, K):
        np.negative(K, out=K)  # make distances positive
        K[K < 0] = 0
        np.sqrt(K, out=K)  # this also takes the root of the length scale
        np.negative(K, out=K)
        np.exp(K, out=K)
        return K


class Delta(IsotropicKernel):
    def K(self, X, X2=None):
        if X2 is None:
            return self.var * np.eye(X.shape[0], order='F')
        else:
            return np.zeros((X.shape[0], X2.shape[0]), order='F')


class Zero(IsotropicKernel):
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return np.zeros((X.shape[0], X2.shape[0]), order='F')

    def Kdiag(self, X):
        return np.zeros(X.shape[0])
