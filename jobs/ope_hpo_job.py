import os
from functools import reduce
from logging import getLogger
from time import time
from pathlib import Path

import numpy as np
import pandas as pd
from obp.dataset import SyntheticBanditDatasetWithActionEmbeds, linear_reward_function
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import OffPolicyEvaluation

from experiments.utils.configs import HpoTrialConfig

from .abstracts.abstract_ope_job import AbstractOpeJob
from .utils.policy import gen_eps_greedy
from .utils.learn_embed import LearnedEmbedMIPS, LearnEmbedMF

logger = getLogger(__name__)

ESTIMATOR = "Learned NMF MIPS"

class OpeHpoJob(AbstractOpeJob):
    def main(self, cfg: HpoTrialConfig):
        logger.info(cfg)
        logger.info(f"The current working directory is {os.getcwd()}")
        start_time = time()
        if not cfg.s3_path.startswith("s3://"):
            Path(cfg.s3_path).mkdir(parents=True, exist_ok=True)

        dataset = SyntheticBanditDatasetWithActionEmbeds(  
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
            random_state=cfg.random_state,
        )
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

        elapsed_prev = 0.0
        squared_error_list = []
        relative_squared_error_list = []
        estimated_policy_value_list = []

        # iterate over n_seeds bootstrap runs
        for t in np.arange(cfg.n_seeds):
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

            ope = OffPolicyEvaluation(
                bandit_feedback=val_bandit_data,
                ope_estimators=[
                    IPS(estimator_name="IPS"),
                    LearnedEmbedMIPS(
                        estimator_name=ESTIMATOR, 
                        n_actions=val_bandit_data["n_actions"], 
                        embed_model=LearnEmbedMF, 
                        learned_embed_params=cfg.learned_embed_params,
                        embed_model_config=cfg.embed_model_config,
                    ),
                ],
            )

            estimated_policy_values = ope.estimate_policy_values(
                action_dist=action_dist_val,
                action_embed=val_bandit_data["action_embed"],
                pi_b=val_bandit_data["pi_b"],
            )
            
            estimated_policy_value_list.append(estimated_policy_values)

            squared_errors = reduce(lambda acc, val: {**acc, val[0]: (val[1] - policy_value) ** 2}, estimated_policy_values.items(), {})
            baseline = squared_errors["IPS"]
            relative_squared_errors = reduce(lambda acc, val: {**acc, val[0]: val[1] / baseline }, squared_errors.items(), {})

            # estimate policy values and calculate MSE of estimators
            squared_error_list.append(squared_errors)
            relative_squared_error_list.append(relative_squared_errors)

            elapsed = np.round((time() - start_time) / 60, 2)
            diff = np.round(elapsed - elapsed_prev, 2)
            logger.info(f"t={t}: {elapsed}min (diff {diff}min)")
            elapsed_prev = elapsed

        # aggregate all results
        value_df = (
            pd.DataFrame(pd.DataFrame(estimated_policy_value_list).stack())
            .reset_index(1)
            .rename(columns={"level_1": "est", 0: "value"})
        )
        value_df.reset_index(inplace=True, drop=True)
        value_df["se"] = (value_df.value - policy_value) ** 2
        value_df["bias"] = 0
        value_df["variance"] = 0
        sample_mean = pd.DataFrame(value_df.groupby(["est"]).mean().value).reset_index()
        for est_ in sample_mean["est"]:
            estimates = value_df.loc[value_df["est"] == est_, "value"].values
            mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
            mean_estimates = np.ones_like(estimates) * mean_estimates
            value_df.loc[value_df["est"] == est_, "bias"] = np.sqrt((
                policy_value - mean_estimates
            ) ** 2)
            value_df.loc[value_df["est"] == est_, "variance"] = np.sqrt((
                estimates - mean_estimates
            ) ** 2)
        value_df.set_index("est", inplace=True)
        value_df.to_csv(f"{cfg.s3_path}/value_df.csv")

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
        mse = rel_result_df.groupby('est').apply(lambda x: np.power(np.e, np.log(x).mean()))['se'][ESTIMATOR]
        lcb = rel_result_df.groupby('est').apply(lambda x: np.power(np.e, np.log(x).mean() - np.log(x).std() / np.sqrt(len(x) - 1)))['se'][ESTIMATOR]
        ucb = rel_result_df.groupby('est').apply(lambda x: np.power(np.e, np.log(x).mean() + np.log(x).std() / np.sqrt(len(x) - 1)))['se'][ESTIMATOR]
        logger.info(f'Relative MSE: {mse:.4f}')
        logger.info(f'Relative MSE LCB: {lcb:.4f}')
        logger.info(f'Relative MSE UCB: {ucb:.4f}')
        logger.info(f'Bias: {value_df["bias"][ESTIMATOR].mean():.10f}')
        logger.info(f'Variance: {value_df["variance"][ESTIMATOR].mean():.10f}')
        