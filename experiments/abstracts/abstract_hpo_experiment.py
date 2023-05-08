import dataclasses
import json
import logging
from enum import Enum
from typing import List

import pandas as pd
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner

from experiments.utils.configs import HpoTrialConfig
from experiments.utils.constants import ROLE

from .abstract_experiments import AbstractExperiment

sm_sess = sagemaker.Session()
sm = sm_sess.sagemaker_client
log = logging.getLogger()


class Status(Enum):
    FAILED = "Failed"
    IN_PROGRESS = "InProgress"


@dataclasses.dataclass
class AbstractHpoExperiment(AbstractExperiment):
    def fit(self, trial_config: HpoTrialConfig, experiment_name):
        job_parameters = self.get_job_config_parameter(trial_config, experiment_name)
        estimator = PyTorch(
            py_version="py38",
            entry_point=self.entry_point,
            role=ROLE,
            sagemaker_session=sagemaker.Session(sagemaker_client=sm),
            framework_version="1.9.0",
            instance_count=1,
            instance_type=self.instance_type,
            hyperparameters=job_parameters,
            metric_definitions=[],
            enable_sagemaker_metrics=True,
            source_dir="./",
            max_run=5 * 24 * 60 * 60  # set the training limit to 5 days (maximum allowed time by default)
        )

        tuner = HyperparameterTuner(
            estimator,
            self.objective_metric_name,
            trial_config.hyperparameter_ranges,
            self.metric_definitions,
            max_jobs=trial_config.max_jobs,
            max_parallel_jobs=self.max_parallel_jobs,
            objective_type=self.objective_type,
            strategy=trial_config.strategy
        )

        training_job_name = f"{experiment_name}-{trial_config.name}"
        trial_name = f"trial-{training_job_name}"
        tuner.fit(
            job_name=training_job_name,
            experiment_config={
                "TrialName": trial_name,
                "TrialComponentDisplayName": "Training",
            },
            wait=False,
        )
        return tuner

    def get_tuner(self, training_job_name):
        return HyperparameterTuner.attach(training_job_name, sagemaker_session=sm_sess)

    def get_job_config_parameter(self, trial: HpoTrialConfig, experiment_name: str) -> dict:
        parameters = {**dataclasses.asdict(trial),
                      **{"s3_path": self.get_s3_path(experiment_name, trial.name),
                         "job_class_name": self.job_class_name}}
        del parameters['hyperparameter_ranges']

        return {"config": f"'{json.dumps(parameters)}'"}

    def get_metrics(self, training_job_name):
        jobs = []

        while True:
            result = sm.list_training_jobs_for_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=training_job_name,
                MaxResults=100,
                SortBy='FinalObjectiveMetricValue',
                SortOrder='Descending'
            )
            jobs += result['TrainingJobSummaries']
            if 'NextToken' not in result:
                break

        result = pd.DataFrame()
        for job in jobs:
            job_description = sm.describe_training_job(TrainingJobName=job['TrainingJobName'])
            metrics = {m['MetricName']: m['Value'] for m in job_description['FinalMetricDataList']}
            result = result.append(pd.Series(metrics, name=job['TrainingJobName']))
        
        return result

    @property
    def metric_definitions(self) -> List[dict]:
        """Metrics for HyperparameterTuner object to extract from logs"""
        return [
            {"Name": "Relative MSE", "Regex": "Relative MSE: ([0-9\\.]+)"},
        ]

    @property
    def objective_metric_name(self) -> str:
        """Metric that tuner optimizes for. Must be specified in the @property metrics_definitions field"""
        return "Relative MSE"

    @property
    def objective_type(self) -> str:
        """Whether the tuner should 'Minimize' or 'Maximize' the objective metric"""
        return "Minimize"

    @property
    def max_parallel_jobs(self) -> int:
        """Maximum number of jobs to run simultaneously"""
        return 8

    @property
    def instance_type(self) -> str:
        """AWS SageMaker Training instance type used for the job execution"""
        return "ml.c5.4xlarge"
