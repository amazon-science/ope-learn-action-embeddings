from logging import getLogger
from pathlib import Path
from typing import List

import pandas as pd

from .abstracts.abstract_experiments import AbstractExperiment
from .utils.configs import SyntheticOpeTrialConfig
from .utils.plots import plot_line


logger = getLogger(__name__)


class NUnobsCatDimExperiment(AbstractExperiment):
    @property
    def job_class_name(self) -> str:
        return "NUnobsCatDimJob"

    @property
    def trial_configs(self) -> List[SyntheticOpeTrialConfig]:
        return [
            SyntheticOpeTrialConfig(
                name="0",
                n_unobserved_cat_dim_list=[0],
                n_cat_dim=20,
            ),
            SyntheticOpeTrialConfig(
                name="2",
                n_unobserved_cat_dim_list=[2],
                n_cat_dim=20,
            ),
            SyntheticOpeTrialConfig(
                name="4",
                n_unobserved_cat_dim_list=[4],
                n_cat_dim=20,
            ),
            SyntheticOpeTrialConfig(
                name="6",
                n_unobserved_cat_dim_list=[6],
                n_cat_dim=20,
            ),
            SyntheticOpeTrialConfig(
                name="8",
                n_unobserved_cat_dim_list=[8],
                n_cat_dim=20,
            ),
            SyntheticOpeTrialConfig(
                name="10",
                n_unobserved_cat_dim_list=[10],
                n_cat_dim=20,
            ),
            SyntheticOpeTrialConfig(
                name="12",
                n_unobserved_cat_dim_list=[12],
                n_cat_dim=20,
            ),
            SyntheticOpeTrialConfig(
                name="14",
                n_unobserved_cat_dim_list=[14],
                n_cat_dim=20,
            ),
            SyntheticOpeTrialConfig(
                name="16",
                n_unobserved_cat_dim_list=[16],
                n_cat_dim=20,
            ),
            SyntheticOpeTrialConfig(
                name="18",
                n_unobserved_cat_dim_list=[18],
                n_cat_dim=20,
            ),
        ]

    @property
    def instance_type(self) -> str:
        return "ml.c5.18xlarge"

    def get_output(self, experiment_name: str, local_path: str = None):
        exclude = [
            # 'MIPS (true)',
            # "Learned MIPS OneHot", 
            # "Learned MIPS FineTune",
            # "Learned MIPS Combined",
        ]
        result = pd.DataFrame()
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
            result_df=result,
            fig_path=output_path,
            x="n_unobserved_cat_dim",
            xlabel="Number of unobserved embedding dimensions",
            xticklabels=result.n_unobserved_cat_dim.unique(),
            exclude=exclude
        )
        if not local_path:
            result.to_csv(f"{self.get_output_path(experiment_name)}/result_df.csv")
