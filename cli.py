import importlib
import logging
import pkgutil
import os

import click

from experiments.abstracts.abstract_experiments import AbstractExperiment

log = logging.getLogger()


@click.group()
def cli():
    pass


@cli.command()
@click.option('--experiment_name', '-n', help='Please provide a unique name')
@click.option('--experiment_class', '-c', help='Please provide experiment class name eg. NActionExperiment')
def run(experiment_name: str, experiment_class: str):
    get_experiment_object(experiment_class).run(experiment_name)
    click.echo("Running Experiment on Sage maker Training Jobs. Please wait until jobs are finished")
    click.echo("Check status here: https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs")


@cli.command()
@click.option('--experiment_name', '-n', help='Please provide the experiment name you set when running the experiment', default=None)
@click.option('--experiment_class', '-c', help='Please provide experiment class name eg. NActionExperiment')
@click.option('--local_path', '-d', help='If the results are stored locally, provide the directory location', default=None)
def output(experiment_name: str, experiment_class: str, local_path: str):
    click.echo("Getting Experiment output on Sage maker Training Jobs.")
    get_experiment_object(experiment_class).get_output(experiment_name, local_path=local_path)
    click.echo("Output has finished successfully")


def get_experiment_object(experiment_class_name: str, relative_import="experiments") -> AbstractExperiment:
    for module in pkgutil.iter_modules([f"{os.path.dirname(os.path.abspath(__file__))}/experiments"]):
        experiment_module = importlib.import_module(f'{relative_import}.{module.name}')
        try:
            experiment_class = getattr(experiment_module, experiment_class_name)
            return experiment_class()
        except AttributeError:
            continue

    raise RuntimeError("could not find experiment class")


if __name__ == '__main__':
    cli()
