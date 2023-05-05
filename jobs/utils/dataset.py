from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from obp.dataset import OpenBanditDataset, SyntheticBanditDatasetWithActionEmbeds
from obp.dataset.reward_type import RewardType
from obp.types import BanditFeedback
from obp.utils import sample_action_fast, softmax
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state, check_scalar
from torch import nn


@dataclass
class ModifiedOpenBanditDataset(OpenBanditDataset):
    """Flattening the list structure of OBD according to item-position click model.
    As OBD has 80 unique actions and 3 different positions in its recommendation interface,
    the resulting action space has the cardinality of 80 * 3 = 240."""
    @property
    def n_actions(self) -> int:
        """Number of actions."""
        return int(self.action.max() + 1)

    def __post_init__(self) -> None:
        """Initialize Open Bandit Dataset Class."""
        if self.behavior_policy not in [
            "bts",
            "random",
        ]:
            raise ValueError(
                f"`behavior_policy` must be either of 'bts' or 'random', but {self.behavior_policy} is given"
            )

        if self.campaign not in [
            "all",
            "men",
            "women",
        ]:
            raise ValueError(
                f"`campaign` must be one of 'all', 'men', or 'women', but {self.campaign} is given"
            )

        self.data_path = f"{self.data_path}/{self.behavior_policy}/{self.campaign}"
        self.raw_data_file = f"{self.campaign}.csv"

        self.load_raw_data()
        self.pre_process()

    def load_raw_data(self) -> None:
        """Load raw open bandit dataset."""
        self.data = pd.read_csv(f"{self.data_path}/{self.raw_data_file}", index_col=0)
        self.item_context = pd.read_csv(
            f"{self.data_path}/item_context.csv", index_col=0
        )
        self.data.sort_values("timestamp", inplace=True)
        self.action = self.data["item_id"].values
        self.position = (rankdata(self.data["position"].values, "dense") - 1).astype(
            int
        )
        self.reward = self.data["click"].values
        self.pscore = self.data["propensity_score"].values

    def pre_process(self) -> None:
        """Preprocess raw open bandit dataset."""
        user_cols = self.data.columns.str.contains("user_feature")
        self.context = pd.get_dummies(
            self.data.loc[:, user_cols], drop_first=True
        ).values
        pos = pd.DataFrame(self.position)
        self.action_context = (
            self.item_context.drop(columns=["item_id", "item_feature_0"], axis=1)
            .apply(LabelEncoder().fit_transform)
            .values
        )
        self.action_context = self.action_context[self.action]
        self.action_context = np.c_[self.action_context, pos]

        self.action = self.position * self.n_actions + self.action
        self.position = np.zeros_like(self.position)
        self.pscore /= 3

    def sample_bootstrap_bandit_feedback(
        self,
        sample_size: Optional[int] = None,
        test_size: float = 0.3,
        is_timeseries_split: bool = False,
        random_state: Optional[int] = None,
    ) -> BanditFeedback:

        if is_timeseries_split:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )[0]
        else:
            bandit_feedback = self.obtain_batch_bandit_feedback(
                test_size=test_size, is_timeseries_split=is_timeseries_split
            )
        n_rounds = bandit_feedback["n_rounds"]
        if sample_size is None:
            sample_size = bandit_feedback["n_rounds"]
        else:
            check_scalar(
                sample_size,
                name="sample_size",
                target_type=(int),
                min_val=0,
                max_val=n_rounds,
            )
        random_ = check_random_state(random_state)
        bootstrap_idx = random_.choice(
            np.arange(n_rounds), size=sample_size, replace=True
        )
        for key_ in [
            "action",
            "position",
            "reward",
            "pscore",
            "context",
            "action_context",
        ]:
            bandit_feedback[key_] = bandit_feedback[key_][bootstrap_idx]
        bandit_feedback["n_rounds"] = sample_size
        return bandit_feedback


class DataGeneratingNetwork(nn.Module):
    def __init__(self, n_features=3, feature_dim=5, context_dim=10, hidden_layer_size=50):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features * feature_dim + context_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, 1)
        )

    def forward(self, context, action_embed):
        x = torch.cat([context, nn.Flatten()(action_embed)], dim=1)
        return self.layers(x)

    def generate_reward(self, context, action_embed, batch_size=32):
        rewards = np.zeros(context.shape[0])
        for i in range(0, context.shape[0], batch_size):
            context_batch = torch.from_numpy(context[i:i + batch_size]).float()
            action_embed_batch = torch.from_numpy(np.tile(action_embed, (context_batch.shape[0], 1, 1))).float()
            rewards_batch = self.forward(context_batch, action_embed_batch)
            rewards[i:i + batch_size] = rewards_batch.detach().numpy().flatten()
        return rewards


@ dataclass
class NonLinearSyntheticBanditDatasetWithActionEmbeds(SyntheticBanditDatasetWithActionEmbeds):
    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)
        self._define_action_embed()
        self.model = DataGeneratingNetwork(n_features=self.n_cat_dim, feature_dim=self.latent_param_mat_dim, context_dim=self.dim_context)

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedback:
        """Obtain batch logged bandit data.

        Parameters
        ----------
        n_rounds: int
            Data size of the synthetic logged bandit data.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Synthesized logged bandit data with action category information.

        """
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        contexts = self.random_.normal(size=(n_rounds, self.dim_context))

        # calc expected rewards given context and action (n_data, n_actions)
        q_x_e = np.zeros((n_rounds, *([self.n_cat_per_dim] * self.n_cat_dim)))
        q_x_a = np.zeros((n_rounds, self.n_actions))

        for index in np.ndindex(q_x_e.shape[1:]):
            embed_param = np.zeros(self.latent_cat_param.shape[::2])
            p_e_a = np.ones(self.n_actions)
            for feature, category in list(zip(range(self.n_cat_dim), index)):
                embed_param[feature] = self.latent_cat_param[feature, category]
                p_e_a *= self.p_e_a[(Ellipsis, category, feature)]

            q_x_e[(Ellipsis, *index)] = self.model.generate_reward(
                context=contexts,
                action_embed=embed_param,
            )            
            # q_x_e[(Ellipsis, *index)] = np.random.randn(n_rounds)
            q_x_a += np.outer(q_x_e[(Ellipsis, *index)], p_e_a)

        # sample actions for each round based on the behavior policy
        if self.behavior_policy_function is None:
            pi_b_logits = q_x_a
        else:
            pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        if self.n_deficient_actions > 0:
            pi_b = np.zeros_like(q_x_a)
            n_supported_actions = self.n_actions - self.n_deficient_actions
            supported_actions = np.argsort(
                self.random_.gumbel(size=(n_rounds, self.n_actions)), axis=1
            )[:, ::-1][:, :n_supported_actions]
            supported_actions_idx = (
                np.tile(np.arange(n_rounds), (n_supported_actions, 1)).T,
                supported_actions,
            )
            pi_b[supported_actions_idx] = softmax(
                self.beta * pi_b_logits[supported_actions_idx]
            )
        else:
            pi_b = softmax(self.beta * pi_b_logits)
        actions = sample_action_fast(pi_b, random_state=self.random_state)

        # sample action embeddings based on sampled actions
        action_embed = np.zeros((n_rounds, self.n_cat_dim), dtype=int)
        for d in np.arange(self.n_cat_dim):
            action_embed[:, d] = sample_action_fast(
                self.p_e_a[actions, :, d],
                random_state=d,
            )

        # sample rewards given the context and action embeddings
        expected_rewards_factual = q_x_e[(np.arange(n_rounds), *action_embed.T)]
        if RewardType(self.reward_type) == RewardType.BINARY:
            rewards = self.random_.binomial(n=1, p=expected_rewards_factual)
        elif RewardType(self.reward_type) == RewardType.CONTINUOUS:
            rewards = self.random_.normal(
                loc=expected_rewards_factual, scale=self.reward_std, size=n_rounds
            )

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            action_context=self.action_context_reg[
                :, self.n_unobserved_cat_dim :
            ].copy(),  # action context used for training a reg model
            action_embed=action_embed[
                :, self.n_unobserved_cat_dim :
            ].copy(),  # action embeddings used for OPE with MIPW
            context=contexts,
            action=actions,
            position=None,  # position effect is not considered in synthetic data
            reward=rewards,
            expected_reward=q_x_a,
            q_x_e=q_x_e[:, :, self.n_unobserved_cat_dim :].copy(),
            p_e_a=self.p_e_a[
                :, :, self.n_unobserved_cat_dim :
            ].copy(),  # true probability distribution of the action embeddings
            pi_b=pi_b[:, :, np.newaxis].copy(),
            pscore=pi_b[np.arange(n_rounds), actions].copy(),
        )
