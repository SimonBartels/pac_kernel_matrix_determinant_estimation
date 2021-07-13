import numpy as np
import inspect
from abc import abstractmethod
import mlflow
from time import thread_time

from kernel.isotropic_kernel import IsotropicKernel

from util.result_processing.result_tables import DefaultTable


class AbstractAlgorithm:
    def __init__(self):
        self.sn2 = None
        self.k = None
        self.logger = mlflow

    @abstractmethod
    def get_signature(self) -> str:
        raise RuntimeError("abstract method!")

    @abstractmethod
    def get_latex_name(self) -> str:
        raise RuntimeError("abstract method!")

    @abstractmethod
    def run_configuration(self, K_, max_iterations, max_time):
        raise RuntimeError("abstract method!")

    def init(self, **kwargs) -> ():
        pass

    def get_results_table_description(self) -> DefaultTable:
        return DefaultTable()

    def set_environment(self, sn2: float, k: IsotropicKernel):
        self.sn2 = sn2
        self.k = k

    def get_parameter_dictionary(self, parsed_arguments):
        d = {}
        args = inspect.signature(self.init).parameters.keys()
        for arg in args:
            #a = arg.replace('_', '-')
            d[arg] = parsed_arguments[self.get_signature() + "_" + arg]
        return d

    def get_default_parameter_dictionary(self):
        d = {}
        parameter_dict = inspect.signature(self.init).parameters
        args = parameter_dict.keys()
        for arg in args:
            #a = arg.replace('_', '-')
            d[arg] = parameter_dict[arg].default
        return d

    def add_parameters(self, parser):
        parameter_dict = inspect.signature(self.init).parameters
        args = parameter_dict.keys()
        for arg in args:
            short = ""
            for l in arg.split('_'):
                short += l[0]
            parser.add_argument("-%s-%s" % (self.get_signature(), short),
                                "--%s-%s" % (self.get_signature(), arg.replace('_', '-')),
                                type=type(parameter_dict[arg].default), default=parameter_dict[arg].default)

    def log_metric(self, *args, **kwargs):
        self.logger.log_metric(*args, **kwargs)

    def log(self, step, estimate, t0):
        self.log_metric(DefaultTable.CUM_TIME, self.time() - t0, step=step)
        self.log_metric(DefaultTable.LOG_DET_ESTIMATE, estimate, step=step)

    def time(self):
        # Note that, process_time() is NOT the right method!
        # The former appears to return the sum of the time spent in each thread!
        return thread_time()

    def _check_result_exists(self, step: int) -> bool:
        if len(mlflow.active_run().data.metrics) == 0:
            return False
        results = mlflow.tracking.MlflowClient().get_metric_history(mlflow.active_run().info.run_id,
                                                                    DefaultTable.LOG_DET_ESTIMATE)
        # ignore nan results -- I'll have to find another way to clean those
        rlist = [r for r in results if r.step == step and not np.isnan(np.array(r.value))]
        l = len(rlist)
        if l == 0:
            return False
        elif l == 1:
            return True
        else:
            raise RuntimeError("There should be only one result for each step!")
