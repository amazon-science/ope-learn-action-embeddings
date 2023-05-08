from logging import getLogger
from pathlib import Path
from typing import List

import pandas as pd

from .abstracts.abstract_experiments import AbstractExperiment
from .utils.configs import SyntheticOpeTrialConfig
from .utils.plots import plot_line


logger = getLogger(__name__)


class NActionsExperiment(AbstractExperiment):
    @property
    def job_class_name(self) -> str:
        return "NActionsJob"

    @property
    def instance_type(self) -> str:
        return "ml.c5.18xlarge"

    @property
    def trial_configs(self) -> List[SyntheticOpeTrialConfig]:
        return [
            SyntheticOpeTrialConfig(
                name="10",
                n_actions_list=[10]
            ),
            SyntheticOpeTrialConfig(
                name="20",
                n_actions_list=[20]
            ),
            SyntheticOpeTrialConfig(
                name="50",
                n_actions_list=[50]
            ),
            SyntheticOpeTrialConfig(
                name="100",
                n_actions_list=[100]
            ),
            SyntheticOpeTrialConfig(
                name="200",
                n_actions_list=[200]
            ),
            SyntheticOpeTrialConfig(
                name="500",
                n_actions_list=[500]
            ),
            SyntheticOpeTrialConfig(
                name="1000",
                n_actions_list=[1000]
            ),
            SyntheticOpeTrialConfig(
                name="2000",
                n_actions_list=[2000]
            ),
        ]

    def get_output(self, experiment_name: str, local_path: str = None):
        result = pd.DataFrame()
        exclude = [
            # 'MIPS (true)',
            # "Learned MIPS OneHot", 
            # "Learned MIPS FineTune",
            # "Learned MIPS Combined",
        ]
        if local_path:
            result = pd.read_csv(f"{local_path}/result_df.csv", index_col=0)
            output_path = Path(local_path)
        else:
            output_path = Path(self.get_output_path(experiment_name))
            for trial in self.trial_configs:
                s3_path = self.get_s3_path(experiment_name, trial.name)
                try:
                    result = result.append(pd.read_csv(f"{s3_path}/result_df.csv", index_col=0), ignore_index=True)
                except:
                    logger.error(f"No result found for {trial.name}")
        plot_line(
            result_df = result,
            fig_path = output_path,
            x = "n_actions",
            xlabel = "Number of actions",
            xticklabels = result.n_actions.unique(),
            exclude = exclude
        )
        if not local_path:
            result.to_csv(f"{self.get_output_path(experiment_name)}/result_df.csv")
