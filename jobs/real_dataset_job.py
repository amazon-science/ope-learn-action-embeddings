import os
from logging import getLogger
from os.path import dirname
from time import time

import numpy as np
import pandas as pd
import torch
from obp.policy import BernoulliTS, Random
from torch.utils.data import DataLoader

from .abstracts.abstract_ope_job import AbstractOpeJob
from .utils.learn_embed import LearnEmbedLinear, TorchBanditDataset
from .utils.ope import run_real_dataset_ope
from .utils.dataset import ModifiedOpenBanditDataset
from experiments.utils.configs import RealOpeTrialConfig

logger = getLogger(__name__)


class RealDatasetJob(AbstractOpeJob):
    def main(self, cfg: RealOpeTrialConfig):
        logger.info(cfg)
        logger.info(f"The current working directory is {os.getcwd()}")
        start_time = time()

        # configurations
        sample_size = cfg.sample_size
        random_state = cfg.random_state
        obd_path = dirname(dirname(dirname(cfg.s3_path))) + "/open_bandit_dataset"
        OBD_N_ACTIONS = 80
        OBD_LEN_LIST = 3

        # define policies
        policy_ur = Random(
            n_actions=OBD_N_ACTIONS,
            len_list=OBD_LEN_LIST,
            random_state=random_state,
        )
        policy_ts = BernoulliTS(
            n_actions=OBD_N_ACTIONS,
            len_list=OBD_LEN_LIST,
            random_state=random_state,
            is_zozotown_prior=True,
            campaign="all",
        )

        # calc ground-truth policy value (on-policy)
        policy_value = ModifiedOpenBanditDataset.calc_on_policy_policy_value_estimate(
            behavior_policy="bts", campaign="all", data_path=obd_path
        )

        # define a dataset class
        dataset = ModifiedOpenBanditDataset(
            behavior_policy="random",
            data_path=obd_path,
            campaign="all",
        )

        elapsed_prev = 0.0
        squared_error_list = []
        relative_squared_error_list = []

        # iterate over n_seeds bootstrap runs
        for t in np.arange(cfg.n_seeds):
            pi_b = policy_ur.compute_batch_action_dist(n_rounds=sample_size)
            pi_e = policy_ts.compute_batch_action_dist(n_rounds=sample_size)
            pi_e = pi_e.reshape(sample_size, OBD_N_ACTIONS * OBD_LEN_LIST, 1) / OBD_LEN_LIST

            val_bandit_data = dataset.sample_bootstrap_bandit_feedback(
                sample_size=sample_size,
                random_state=t,
            )
            val_bandit_data["pi_b"] = pi_b.reshape(sample_size, OBD_N_ACTIONS * OBD_LEN_LIST, 1) / OBD_LEN_LIST

            # learn the reward model for DM and DR methods - same model as Learned MIPS OneHot
            model = LearnEmbedLinear(
                action_dim=val_bandit_data['action_context'].shape[1],
                action_cat_dim=len(np.unique(val_bandit_data['action_context'])),
                n_actions=val_bandit_data['n_actions'],
                context_dim=val_bandit_data['context'].shape[1],
                config=cfg.embed_model_config
            )
            
            model_dataset = TorchBanditDataset(
                n_actions=val_bandit_data['n_actions'],
                n_dim_action=len(np.unique(val_bandit_data['action_context'])), 
                context=val_bandit_data['context'], 
                action=val_bandit_data['action'], 
                action_embed=val_bandit_data['action_context'], 
                reward=val_bandit_data['reward']
            )
            train_dataloader = DataLoader(model_dataset, batch_size=32, shuffle=True)

            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for _ in range(cfg.learned_embed_params.epochs):
                model.train_loop(train_dataloader, loss_fn, optimizer)

            estimated_rewards = np.zeros((val_bandit_data['n_rounds'], val_bandit_data['n_actions'], 1))
            for i in range(val_bandit_data['n_actions']):
                indices = np.where(dataset.action == i)
                if len(indices) != 0:
                    _, action, action_embed, _ = model_dataset.__getitem__(i)
                else:
                    action_embed = np.zeros(val_bandit_data['action_context'].shape[1]).astype(int)
                estimated_rewards[:,i,:] = model(
                    torch.from_numpy(val_bandit_data['context']).float(),
                    action.repeat(val_bandit_data['n_rounds'], 1),
                    action_embed.repeat(val_bandit_data['n_rounds'], 1, 1)
                ).detach().numpy()     

            estimated_rewards_dict = {
                "DR": estimated_rewards,
                "DM": estimated_rewards,
                "SwitchDR": estimated_rewards,
            }

            # estimate policy values and calculate MSE of estimators
            squared_errors, relative_squared_errors = run_real_dataset_ope(
                val_bandit_data=val_bandit_data,
                action_dist_val=pi_e,
                estimated_rewards=estimated_rewards_dict,
                policy_value=policy_value,
                embed_model_config=cfg.embed_model_config,
                learned_embed_params=cfg.learned_embed_params,
                logging_losses_file = f"{cfg.s3_path}/model_losses/{time()}.parquet"
            )
            squared_error_list.append(squared_errors)
            relative_squared_error_list.append(relative_squared_errors)

            elapsed = np.round((time() - start_time) / 60, 2)
            diff = np.round(elapsed - elapsed_prev, 2)
            logger.info(f"t={t}: {elapsed}min (diff {diff}min)")
            elapsed_prev = elapsed

            # aggregate all results
            result_df = (
                pd.DataFrame(pd.DataFrame(squared_error_list).stack())
                .reset_index(1)
                .rename(columns={"level_1": "est", 0: "se"})
            )
            result_df.reset_index(inplace=True, drop=True)
            result_df.to_csv(f"{cfg.s3_path}/result_df.csv")

        rel_result_df = (
            pd.DataFrame(pd.DataFrame(relative_squared_error_list).stack())
            .reset_index(1)
            .rename(columns={"level_1": "est", 0: "se"})
        )
        rel_result_df.reset_index(inplace=True, drop=True)
        rel_result_df.to_csv(f"{cfg.s3_path}/rel_result_df.csv")
