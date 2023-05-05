import argparse
import importlib
import json
import os
import pkgutil
from logging import getLogger
from types import SimpleNamespace

from jobs.abstracts.abstract_job import AbstractJob


logger = getLogger(__name__)

def get_job_object(job_class_name: str, relative_import="jobs") -> AbstractJob:
    for module in pkgutil.iter_modules([f"{os.path.dirname(os.path.abspath(__file__))}/jobs"]):
        job_module = importlib.import_module(f"{relative_import}.{module.name}")
        try:
            job_class = getattr(job_module, job_class_name)
            return job_class()
        except AttributeError:
            continue

    raise RuntimeError("could not find job class")


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        logger.info(f"Unknown arguments passed: {unknown}")
    cfg = json.loads(args.config, object_hook=lambda d: SimpleNamespace(**d))
    get_job_object(cfg.job_class_name).main(cfg)


if __name__ == "__main__":
    init()
