import abc
import argparse
import dataclasses
from abc import ABC, ABCMeta
from logging import getLogger


from experiments.utils.configs import LearnedEmbedParams, LearnEmbedConfig, SyntheticOpeTrialConfig


logger = getLogger(__name__)

class HandleOpeArgs(ABCMeta):
    def __init__(cls, name, bases, clsdict):
        if 'main' in clsdict:
            def ope_args_main(self, cfg: SyntheticOpeTrialConfig):
                parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

                parser.add_argument("--batch-size", type=int)
                parser.add_argument("--epochs", type=int)
                parser.add_argument("--lr", type=float)

                # See LearnEmbedConfig for the description of the parameters
                parser.add_argument("--learned-embed-dim", type=int)

                args, unknown = parser.parse_known_args()
                logger.info(f"Uknown args: {unknown}")

                cfg.embed_model_config = LearnEmbedConfig(**{
                    **cfg.embed_model_config.__dict__,
                    **dict([(key, value) for key, value in vars(args).items() if key in vars(LearnEmbedConfig()).keys()])
                })
                cfg.learned_embed_params = LearnedEmbedParams(**{
                    **cfg.learned_embed_params.__dict__,
                    **dict([(key, value) for key, value in vars(args).items() if key in vars(LearnedEmbedParams()).keys()])
                })

                clsdict['main'](self, cfg)
            setattr(cls, 'main', ope_args_main)


@dataclasses.dataclass
class AbstractJob(ABC, metaclass=HandleOpeArgs):

    @abc.abstractmethod
    def main(self, cfg):
        """
        This the main compute function for the training job.
        @param cfg: config object that contains all parameters passed to the job from the experiment
        """
        pass
