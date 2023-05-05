from dataclasses import dataclass, field
from typing import List


@dataclass
class TrialConfig:
    name: str # name of the trial


@dataclass
class LearnEmbedConfig:
    learned_embed_dim: int = 5 # size of the learned action embedding


@dataclass
class LearnedEmbedParams:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    train_test_ratio: float = 1


@dataclass
class OpeTrialConfig(TrialConfig):
    embed_model_config: LearnEmbedConfig = field(default_factory=lambda: LearnEmbedConfig())
    learned_embed_params: LearnedEmbedParams = field(default_factory=lambda: LearnedEmbedParams())


@dataclass
class SyntheticOpeTrialConfig(OpeTrialConfig):
    n_cat_dim: int = 3 # number of dimensions in the action embedding
    n_unobserved_cat_dim: int = 0 # number of unobserved dimensions in the action embedding
    dim_context: int = 10 # number of context dimensions
    n_seeds: int = 100 # number of runs to average over
    n_val_data: int = 10000 # default number of training samples
    n_actions: int = 100 # default number of distinct actions
    n_cat_per_dim: int = 10 # number of categories per dimension in the action embedding
    n_test_data: int = 200000 # number of test samples
    n_def_actions: int = 0 # number of actions in which we do not have any observations
    latent_param_mat_dim: int = 5 # size of the random parameters matrix to generate the reward
    beta: int = -1 # entropy of the logging policy, -1 means almost random uniform, 1 means almost deterministic
    eps: float = 0.05 # the amount of exploration in eps-greedy evaluation policy
    reward_std: float = 2.5 # amount of gaussian noise in the reward
    is_optimal: bool = True # whether the policy selects the best or the worst action
    embed_selection: bool = False # whether to use the SLOPE algorithm for embedding selection
    random_state: int = 12345 # fixed seed to replicate the same results
    n_val_data_list: List[int] = field(default_factory=lambda: [800, 1600, 3200, 6400, 12800, 25600]) # values when varying the number of training samples
    n_unobserved_cat_dim_list: List[int] = field(
        default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]) # values when varying the number of unobserved dimensions in the action embedding
    eps_list: List[float] = field(default_factory=lambda: [0, 0.2, 0.4, 0.6, 0.8, 1]) # values when varying the amount of exploration in eps-greedy evaluation policy
    beta_list: List[float] = field(default_factory=lambda: [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]) # values when varying the entropy of the logging policy
    noise_list: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]) # values when varying the amount of gaussian noise in the reward
    n_def_actions_list: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]) # values when varying the number of actions in which we do not have any observations
    n_actions_list: List[float] = field(
        default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    ) # values when varying the number of distinct actions


@dataclass
class RealOpeTrialConfig(OpeTrialConfig):
    n_seeds: int = 150 # number of bootstrap runs
    sample_size: int = 1000 # number of observations in one sample
    random_state: str = 12345 # fixed seed to replicate the same results


@dataclass
class HpoTrialConfig(SyntheticOpeTrialConfig):
    random_state: str = 12345 # fixed seed to replicate the same results
    hyperparameter_ranges: dict = field(default_factory=lambda: {}) # hyperparameter space defined by sagemaker hyperparameter ranges https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-ranges.html
    fixed_hyperparameters: dict = field(default_factory=lambda: {}) # fixed values for already optimized hyperparameters, ignored if present in hyperparameter_ranges
    max_jobs: int = 100 # number of maximum trials when searching for the hyperparameters
    strategy: str = "Bayesian" # strategy used to search over the hyperparameter space. Either 'Bayesian' or 'Random'
