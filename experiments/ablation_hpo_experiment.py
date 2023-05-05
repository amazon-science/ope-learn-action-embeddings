from logging import getLogger
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
from sagemaker.tuner import IntegerParameter

from .abstracts.abstract_hpo_experiment import AbstractHpoExperiment
from .utils.configs import HpoTrialConfig

plt.style.use(['science', 'no-latex'])
logger = getLogger(__name__)


class AblationHpoExperiment(AbstractHpoExperiment):
    @property
    def job_class_name(self) -> str:
        return "OpeHpoJob"

    @property
    def max_parallel_jobs(self) -> int:
        return 5

    @property
    def instance_type(self) -> str:
        return "ml.c5.18xlarge"

    @property
    def trial_configs(self) -> List[HpoTrialConfig]:
        return [
            HpoTrialConfig(
                name="learned-embed-dim",
                hyperparameter_ranges={
                    "learned-embed-dim": IntegerParameter(1, 100),
                },
                max_jobs=100
            ),
        ]

    @property
    def metric_definitions(self) -> List[dict]:
        """Metrics for HyperparameterTuner object to extract from logs"""
        return [
            {"Name": "Relative MSE", "Regex": "Relative MSE: ([0-9\\.]+)"},
            {"Name": "Relative MSE LCB", "Regex": "Relative MSE LCB: ([0-9\\.]+)"},
            {"Name": "Relative MSE UCB", "Regex": "Relative MSE UCB: ([0-9\\.]+)"},
            {"Name": "Bias", "Regex": "Bias: ([0-9\\.]+)"},
            {"Name": "Variance", "Regex": "Variance: ([0-9\\.]+)"},
        ]

    def get_output(self, experiment_name: str):
        for trial in self.trial_configs:
            try:
                job_name = f"{experiment_name}-{trial.name}"
                tuner = self.get_tuner(job_name)
            except:
                logger.error(f"Tuning job {experiment_name}-{trial.name} not found")
                continue
            df = tuner.analytics().dataframe()
            df = df.join(self.get_metrics(job_name), on="TrainingJobName")
            hyperparameter_name = list(trial.hyperparameter_ranges.keys())[0]
            df[hyperparameter_name] = df[hyperparameter_name].astype(str).str.replace('"', '').astype(float)
            df.sort_values(by=hyperparameter_name, inplace=True)
            ax = sns.lineplot(
                x=df[hyperparameter_name].rename("Size of the learned embeddings"),
                y=df['FinalObjectiveValue'].rename(self.objective_metric_name)
            )
            ax.fill_between(df[hyperparameter_name], df["Relative MSE LCB"], df["Relative MSE UCB"], alpha=0.2)

            plt.yscale('log')
            plt.ylim(top=5, bottom=4e-1)
            output_dir = self.get_output_path(experiment_name, trial.name)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            tuner.analytics().export_csv(f"{output_dir}/tuner_results.csv")
            plt.savefig(f"{output_dir}/{hyperparameter_name}.pdf", bbox_inches='tight')
            plt.savefig(f"{output_dir}/{hyperparameter_name}.png", bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
            sns.lineplot(
                x=df[hyperparameter_name].rename("Size of the learned embeddings"),
                y=df['Variance'],
                ax=ax[0]
            )
            sns.lineplot(
                x=df[hyperparameter_name].rename("Size of the learned embeddings"),
                y=df['Bias'],
                ax=ax[1]
            )
            plt.savefig(f"{output_dir}/{hyperparameter_name}_bias-variance.pdf", bbox_inches='tight')
            plt.savefig(f"{output_dir}/{hyperparameter_name}_bias-variance.png", bbox_inches='tight')
            plt.close()
