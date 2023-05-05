from logging import getLogger
from pathlib import Path
from typing import List

import pandas as pd

from .abstracts.abstract_experiments import AbstractExperiment
from .utils.configs import SyntheticOpeTrialConfig
from .utils.plots import plot_line


logger = getLogger(__name__)


class NValDataExperiment(AbstractExperiment):
    @property
    def job_class_name(self) -> str:
        return "NValDataJob"

    @property
    def trial_configs(self) -> List[SyntheticOpeTrialConfig]:
        return [
            SyntheticOpeTrialConfig(
                name="800",
                n_val_data_list=[800]
            ),
            SyntheticOpeTrialConfig(
                name="1600",
                n_val_data_list=[1600]
            ),
            SyntheticOpeTrialConfig(
                name="3200",
                n_val_data_list=[3200]
            ),
            SyntheticOpeTrialConfig(
                name="6400",
                n_val_data_list=[6400]
            ),
            SyntheticOpeTrialConfig(
                name="12800",
                n_val_data_list=[12800]
            ),
            SyntheticOpeTrialConfig(
                name="25600",
                n_val_data_list=[25600]
            ),
            SyntheticOpeTrialConfig(
                name="51200",
                n_val_data_list=[51200]
            ),
            SyntheticOpeTrialConfig(
                name="102400",
                n_val_data_list=[102400]
            ),
        ]

    @property
    def instance_type(self) -> str:
        return "ml.c5.18xlarge"

    def get_output(self, experiment_name: str):
        result = pd.DataFrame()
        exclude = [
            # 'MIPS (true)',
            # "Learned MIPS OneHot", 
            # "Learned MIPS FineTune",
            # "Learned MIPS Combined",
        ]
        for trial in self.trial_configs:
            s3_path = self.get_s3_path(experiment_name, trial.name)
            try:
                result = result.append(pd.read_csv(f"{s3_path}/result_df.csv", index_col=0), ignore_index=True)
            except:
                logger.error(f"No result found for {trial.name}")
        plot_line(
            result_df=result,
            fig_path=Path(self.get_output_path(experiment_name)),
            x="n_val_data",
            xlabel="Number of training samples",
            xticklabels=result.n_val_data.unique(),
            exclude=exclude
        )
        result.to_csv(f"{self.get_output_path(experiment_name)}/result_df.csv")
