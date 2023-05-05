import abc
import dataclasses
import json
import logging
from abc import ABC
import time
from enum import Enum
from typing import List

import sagemaker
from sagemaker.pytorch import PyTorch
from smexperiments import tracker, experiment
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent

from experiments.utils.configs import TrialConfig
from experiments.utils.constants import ROLE, BUCKET

sm_sess = sagemaker.Session()
sm = sm_sess.sagemaker_client
log = logging.getLogger()


class Status(Enum):
    FAILED = "Failed"
    IN_PROGRESS = "InProgress"


@dataclasses.dataclass
class AbstractExperiment(ABC):

    def run(self, experiment_name: str):
        """
        All experiments should extend this abstract class and implement
        all abstract functions
        @param experiment_name: unique experiment name
        """
        experiment = Experiment.create(
            experiment_name=experiment_name,
            description=self.description,
            sagemaker_boto_client=sm,
        )
        estimators = []
        for trial_config in self.trial_configs:
            job_parameters = self.get_job_config_parameter(trial_config, experiment_name)
            trial_name = f"trial-{experiment_name}-{trial_config.name}"
            log.info(trial_name)
            sm_trial = Trial.create(
                trial_name=trial_name,
                experiment_name=experiment.experiment_name,
                sagemaker_boto_client=sm,
            )

            log.info(job_parameters)
            with tracker.Tracker.create() as trail_tracker:
                trail_tracker.log_parameters({**job_parameters})
                sm_trial.add_trial_component(trail_tracker)

            estimator = self.fit(trial_config, experiment_name)
            estimators.append(estimator)
        return estimators

    def fit(self, trial_config, experiment_name):
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
            source_dir="src/",
            max_run=5 * 24 * 60 * 60  # set the training limit to 5 days (maximum allowed time by default)
        )

        training_job_name = f"{experiment_name}-{trial_config.name}"
        trial_name = f"trial-{training_job_name}"
        estimator.fit(
            job_name=training_job_name,
            experiment_config={
                "TrialName": trial_name,
                "TrialComponentDisplayName": "Training",
            },
            wait=False,
        )
        return estimator

    @staticmethod
    def training_jobs_has_finished(experiment_name) -> bool:
        sm_experiment = experiment.Experiment.load(experiment_name)
        jobs = []
        for sm_trial_summary in sm_experiment.list_trials():
            for trial_component_summary in Trial.load(sm_trial_summary.trial_name).list_trial_components():
                trial_component = TrialComponent.load(trial_component_summary.trial_component_name)
                if "sagemaker_job_name" in trial_component.parameters:
                    job_name = trial_component.parameters["sagemaker_job_name"].replace("'", "").replace('"', "")
                    jobs.append(sm_sess.describe_training_job(job_name))
        [log.info((job["TrainingJobName"], job["TrainingJobStatus"])) for job in jobs]
        failed_jobs = [job for job in jobs if job["TrainingJobStatus"] == Status.FAILED.value]
        in_progress_jobs = [job for job in jobs if job["TrainingJobStatus"] == Status.IN_PROGRESS.value]

        if failed_jobs:
            log.info("Jobs have failed")
            return False

        if in_progress_jobs:
            log.info("Jobs are in progress")
            return False

        return True

    @abc.abstractmethod
    def get_output(self, experiment_name: str):
        """
        Generates output of the experiment from the data published
        to the s3 bucket, graphs or tables can be the output of an experiment
        @param experiment_name: The name of the experiment which the output is related to
        """
        pass

    def get_s3_path(self, experiment_name, trial_name=None) -> str:
        result = f"s3://{BUCKET}/experiments/experiment={experiment_name}"
        return result if not trial_name else result + f"/trial={trial_name}"

    def list_s3_files(self, experiment_name, trial_name) -> List[str]:
        objects = []
        request_params = {}
        while True:
            response = sm_sess.boto_session.client('s3').list_objects_v2(
                Bucket=BUCKET, Prefix=f"experiments/experiment={experiment_name}/trial={trial_name}", **request_params
            )
            objects += response["Contents"]
            if "NextContinuationToken" in response:
                request_params["ContinuationToken"] = response["NextContinuationToken"]
            else:
                break

        return [f"s3://{BUCKET}/{obj['Key']}" for obj in objects]

    def get_output_path(self, experiment_name, trial_name=None) -> str:
        result = f"./log/{experiment_name}"
        return result if not trial_name else result + f"/{trial_name}"

    def get_job_config_parameter(self, trial: TrialConfig, experiment_name: str) -> dict:
        parameters = {**dataclasses.asdict(trial),
                      **{"s3_path": self.get_s3_path(experiment_name, trial.name),
                         "job_class_name": self.job_class_name}}

        return {"config": f"'{json.dumps(parameters)}'"}

    @property
    @abc.abstractmethod
    def trial_configs(self) -> List[TrialConfig]:
        """
        This function returns the trial configs for each training job,
        it needs to be overridden by each subclass
        """
        pass

    @property
    @abc.abstractmethod
    def job_class_name(self) -> str:
        """
        This function returns the training job class name e.g ActionsJob,
        it needs to be overridden by each subclass
        """
        pass

    @property
    def entry_point(self) -> str:
        return "entry_point.py"

    @property
    def instance_type(self) -> str:
        return "ml.c5.xlarge"

    @property
    def description(self) -> str:
        """
        This function returns the training job class name e.g ActionsJob,
        it needs to be overridden by each subclass
        """
        return " "
