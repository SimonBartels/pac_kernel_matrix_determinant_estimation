import numpy as np
from gpytorch.lazy import DiagLazyTensor, NonLazyTensor
from torch import tensor, float64
from gpytorch.functions import logdet
import gpytorch.settings as gpt_settings
from gpytorch.settings import num_trace_samples, cg_tolerance, max_cg_iterations, max_preconditioner_size
from gpytorch.lazy.added_diag_lazy_tensor import AddedDiagLazyTensor

from AbstractAlgorithm import AbstractAlgorithm
from util.result_processing.result_tables import DefaultTable


class GPyTorch(AbstractAlgorithm):
    def init(self, num_trace_samples=None, cg_tolerance=None, preconditioning_steps=None):
        if num_trace_samples is None:
            num_trace_samples = gpt_settings.num_trace_samples.value()
        if cg_tolerance is None:
            cg_tolerance = gpt_settings.cg_tolerance.value()
        if preconditioning_steps is None:
            preconditioning_steps = gpt_settings.max_preconditioner_size.value()
        self.num_trace_samples = num_trace_samples
        self.cg_tolerance = cg_tolerance
        self.preconditioning_steps = preconditioning_steps

    def get_signature(self):
        return "gpt"

    def run_configuration(self, K_, max_iterations, max_time):
        np.fill_diagonal(K_, K_[0, 0] - self.sn2)
        K = AddedDiagLazyTensor(NonLazyTensor(tensor(K_, dtype=float64)),
                                DiagLazyTensor(tensor(self.sn2 * np.ones(K_.shape[0]), dtype=float64)))
        step = min(self.preconditioning_steps+max_iterations, K_.shape[0])
        t = self.time()
        with num_trace_samples(self.num_trace_samples), cg_tolerance(self.cg_tolerance), \
             max_cg_iterations(max_iterations), \
             max_preconditioner_size(self.preconditioning_steps):
            d = logdet(K)
        t = self.time() - t
        self.log_metric(DefaultTable.CUM_TIME, t, step=step)
        self.log_metric(DefaultTable.LOG_DET_ESTIMATE, d.item(), step=step)
