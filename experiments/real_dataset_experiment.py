from typing import List
from pathlib import Path

import pandas as pd


from .utils.configs import RealOpeTrialConfig
from .abstracts.abstract_experiments import AbstractExperiment
from .utils.plots import plot_cdf


class RealDatasetExperiment(AbstractExperiment):
    @property
    def job_class_name(self) -> str:
        return "RealDatasetJob"

    @property
    def trial_configs(self) -> List[RealOpeTrialConfig]:
        return [
            RealOpeTrialConfig(
                name="1000",
                sample_size=1000,
            ),
            RealOpeTrialConfig(
                name="10000",
                sample_size=10000,
            ),
            RealOpeTrialConfig(
                name="50000",
                sample_size=50000,
            ),
            RealOpeTrialConfig(
                name="100000",
                sample_size=100000,
            )
        ]

    @property
    def instance_type(self) -> str:
        return "ml.c5.18xlarge"

    def get_output(self, experiment_name: str, local_path: str = None):
        exclude = [
            # "Learned MIPS OneHot", 
            # "Learned MIPS FineTune",
            # "Learned MIPS Combined",
            # "SNIPS",
            # "SwitchDR",
        ]
        for trial in self.trial_configs:
            s3_path = self.get_s3_path(experiment_name, trial.name) if not local_path else f"{local_path}/{trial.name}"
            fig_dir = self.get_output_path(experiment_name, trial.name) if not local_path else f"{local_path}/{trial.name}"
            Path(fig_dir).mkdir(parents=True, exist_ok=True)
            result_df = pd.read_csv(f"{s3_path}/result_df.csv")
            result_df.to_csv(f"{fig_dir}/result_df.csv")
            plot_cdf(
                result_df=result_df,
                fig_path=f"{fig_dir}/cdf_IPS.png",
                relative_to="IPS",
                exclude=exclude
            )
            plot_cdf(
                result_df=result_df,
                fig_path=f"{fig_dir}/cdf_MIPS.png",
                exclude=exclude
            )
            plot_cdf(
                result_df=result_df,
                fig_path=f"{fig_dir}/cdf_onehot.png",
                relative_to="Learned MIPS OneHot",
                exclude=exclude,
            )
            plot_cdf(
                result_df=result_df,
                fig_path=f"{fig_dir}/cdf_IPS.pdf",
                relative_to="IPS",
                exclude=exclude,
                remove_legend=True
            )
