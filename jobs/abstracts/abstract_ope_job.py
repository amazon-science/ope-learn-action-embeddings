import abc
import dataclasses
from logging import getLogger

from experiments.utils.configs import OpeTrialConfig

from .abstract_job import AbstractJob


@dataclasses.dataclass
class AbstractOpeJob(AbstractJob):
    @abc.abstractmethod
    def main(self, cfg: OpeTrialConfig):
        """
        This the main compute function for the training job.
        @param cfg: config object that contains all parameters passed to the job from the experiment
        """
        pass
