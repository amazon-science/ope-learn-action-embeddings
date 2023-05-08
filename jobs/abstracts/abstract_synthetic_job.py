import abc
import warnings
from logging import getLogger
from pathlib import Path
from time import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from obp.dataset import SyntheticBanditDatasetWithActionEmbeds, linear_reward_function
from pandas import DataFrame
from sklearn.exceptions import ConvergenceWarning
from torch.utils.data import DataLoader

from experiments.utils.configs import SyntheticOpeTrialConfig

from ..utils.ope import run_ope
from ..utils.learn_embed import LearnEmbedLinear, TorchBanditDataset
from ..utils.policy import gen_eps_greedy
from ..utils.dataset import NonLinearSyntheticBanditDatasetWithActionEmbeds
from .abstract_ope_job import AbstractOpeJob

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = getLogger(__name__)


class AbstractSyntheticJob(AbstractOpeJob):
    @abc.abstractmethod
    def main(self, cfg: SyntheticOpeTrialConfig):
        pass

    def run(self, cfg, hyperparam_name, hyperparam_list):
        logger.info(f"The current working directory is {Path().cwd()}")
        if not cfg.s3_path.startswith("s3://"):
            Path(cfg.s3_path).mkdir(parents=True, exist_ok=True)
        start_time = time()

        # log path
        random_state = cfg.random_state

        elapsed_prev = 0.0
        result_df_list = []
        for hyperparam in hyperparam_list:
            setattr(cfg, hyperparam_name, hyperparam)

            estimated_policy_value_list = []
            # define a dataset class
            dataset = SyntheticBanditDatasetWithActionEmbeds(
            # dataset = NonLinearSyntheticBanditDatasetWithActionEmbeds(
                n_actions=cfg.n_actions,
                dim_context=cfg.dim_context,
                beta=cfg.beta,
                reward_type="continuous",
                n_cat_per_dim=cfg.n_cat_per_dim,
                latent_param_mat_dim=cfg.latent_param_mat_dim,
                n_cat_dim=cfg.n_cat_dim,
                n_unobserved_cat_dim=cfg.n_unobserved_cat_dim,
                n_deficient_actions=int(cfg.n_actions * cfg.n_def_actions),
                reward_function=linear_reward_function,
                reward_std=cfg.reward_std,
                random_state=random_state,
            )
            # test bandit data is used to approximate the ground-truth policy value
            test_bandit_data = dataset.obtain_batch_bandit_feedback(
                n_rounds=cfg.n_test_data
            )
            action_dist_test = gen_eps_greedy(
                expected_reward=test_bandit_data["expected_reward"],
                is_optimal=cfg.is_optimal,
                eps=cfg.eps,
            )
            policy_value = dataset.calc_ground_truth_policy_value(
                expected_reward=test_bandit_data["expected_reward"],
                action_dist=action_dist_test,
            )

            for t in range(cfg.n_seeds):
                # generate validation data
                val_bandit_data = dataset.obtain_batch_bandit_feedback(
                    n_rounds=cfg.n_val_data,
                )

                # make decisions on validation data
                action_dist_val = gen_eps_greedy(
                    expected_reward=val_bandit_data["expected_reward"],
                    is_optimal=cfg.is_optimal,
                    eps=cfg.eps,
                )

                # Direct Method
                model = LearnEmbedLinear(
                    action_dim=val_bandit_data['action_context'].shape[1],
                    action_cat_dim=len(np.unique(val_bandit_data['action_embed'])),
                    n_actions=val_bandit_data['n_actions'],
                    context_dim=val_bandit_data['context'].shape[1],
                    config=cfg.embed_model_config
                )

                model_dataset = TorchBanditDataset(
                    n_actions=val_bandit_data['n_actions'],
                    n_dim_action=len(np.unique(val_bandit_data['action_embed'])),
                    context=val_bandit_data['context'],
                    action=val_bandit_data['action'],
                    action_embed=val_bandit_data['action_embed'],
                    reward=val_bandit_data['reward']
                )
                train_dataloader = DataLoader(model_dataset, batch_size=32, shuffle=True)

                loss_fn = torch.nn.MSELoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                for _ in range(cfg.learned_embed_params.epochs):
                    model.train_loop(train_dataloader, loss_fn, optimizer)

                estimated_rewards = np.zeros((val_bandit_data['n_rounds'], val_bandit_data['n_actions'], 1))
                for i in range(val_bandit_data['n_actions']):
                    _, action, action_embed, _ = model_dataset.__getitem__(i)
                    estimated_rewards[:, i, :] = model(
                        torch.from_numpy(val_bandit_data['context']).float(),
                        action.repeat(val_bandit_data['n_rounds'], 1),
                        action_embed.repeat(val_bandit_data['n_rounds'], 1, 1)
                    ).detach().numpy() 

                estimated_rewards_dict = {
                    "DM": estimated_rewards,
                    "DR": estimated_rewards,
                }

                estimated_policy_values = run_ope(
                    val_bandit_data=val_bandit_data,
                    action_dist_val=action_dist_val,
                    estimated_rewards=estimated_rewards_dict,
                    embed_model_config=cfg.embed_model_config,
                    learned_embed_params=cfg.learned_embed_params,
                    logging_losses_file=f"{cfg.s3_path}/model_losses/{time()}.parquet"
                )
                estimated_policy_value_list.append(estimated_policy_values)
                elapsed = np.round((time() - start_time) / 60, 2)
                diff = np.round(elapsed - elapsed_prev, 2)
                logger.info(f"t={t}: {elapsed}min (diff {diff}min)")
                elapsed_prev = elapsed

                # summarize results
                result_df = (
                    DataFrame(DataFrame(estimated_policy_value_list).stack())
                    .reset_index(1)
                    .rename(columns={"level_1": "est", 0: "value"})
                )
                result_df[hyperparam_name] = hyperparam
                result_df["se"] = (result_df.value - policy_value) ** 2
                result_df["bias"] = 0
                result_df["variance"] = 0
                sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
                for est_ in sample_mean["est"]:
                    estimates = result_df.loc[result_df["est"] == est_, "value"].values
                    mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
                    mean_estimates = np.ones_like(estimates) * mean_estimates
                    result_df.loc[result_df["est"] == est_, "bias"] = (
                        policy_value - mean_estimates
                    ) ** 2
                    result_df.loc[result_df["est"] == est_, "variance"] = (
                        estimates - mean_estimates
                    ) ** 2
                result_df_list.append(result_df)

                # aggregate all results
                result_df = pd.concat(result_df_list).reset_index(level=0)
                result_df.to_csv(f"{cfg.s3_path}/result_df.csv")
        return result_df
