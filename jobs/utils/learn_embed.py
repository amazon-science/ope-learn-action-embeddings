from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from obp.ope import MarginalizedInverseProbabilityWeighting
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from experiments.utils.configs import LearnEmbedConfig, LearnedEmbedParams


def disable_training(f):
    def wrapped(self, *args, **kwargs):
        self.action_embed_stack.training = False
        output = f(self, *args, **kwargs)
        self.action_embed_stack.training = True
        return output

    return wrapped


class LearnEmbedNetwork(nn.Module):
    def __init__(self, action_dim=3, n_actions=10, action_cat_dim=10, context_dim=10, config=LearnEmbedConfig()):
        super(LearnEmbedNetwork, self).__init__()

    def action_embeddings(self, action=None, action_embed=None):
        """Returns learned action embeddings for the given action and/or pre-defined action embedding"""
        return NotImplementedError

    def train_loop(self, dataloader, loss_fn, optimizer):
        for x, a, e, r in dataloader:
            # Compute prediction and loss
            pred = self(x, a, e)
            loss = loss_fn(pred, r)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def report_loss(self, dataloader, loss_fn):
        loss = 0
        num_batches = len(dataloader)
        self.eval()
        for x, a, e, r in dataloader:
            pred = self(x, a, e)
            loss += loss_fn(pred, r).item()
        self.train()
        return loss / num_batches


class LearnEmbedLinear(LearnEmbedNetwork):
    """Linear learning objective, where the action network only uses the action identity"""

    def __init__(self, action_dim=3, n_actions=10, action_cat_dim=10, context_dim=10, config=LearnEmbedConfig()):
        super().__init__()
        self.action_embed_stack = nn.Linear(n_actions, context_dim)
        self.output = nn.Linear(context_dim, 1)
        nn.init.ones_(self.output.weight)
        self.output.requires_grad_(False)

    def forward(self, context, action, action_embed):
        x1 = self.action_embed_stack(action)
        out = self.output(torch.mul(x1, context))
        return out

    @disable_training
    def action_embeddings(self, action=None, action_embed=None):
        return self.action_embed_stack(action)



class LearnEmbedLinearB(LearnEmbedNetwork):
    """Linear learning objective, where the action network uses the action embeddings"""

    def __init__(self, action_dim=3, n_actions=10, action_cat_dim=10, context_dim=10, config=LearnEmbedConfig()):
        super().__init__()
        self.action_embed_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(action_cat_dim * action_dim, context_dim)
        )
        self.output = nn.Linear(context_dim, 1)
        nn.init.ones_(self.output.weight)
        self.output.requires_grad_(False)

    def forward(self, context, action, action_embed):
        x1 = self.action_embed_stack(action_embed)
        out = self.output(torch.mul(x1, context))
        return out

    @disable_training
    def action_embeddings(self, action=None, action_embed=None):
        return self.action_embed_stack(action_embed)



class LearnEmbedLinearC(LearnEmbedNetwork):
    """Linear learning objective, where the action network only uses both the action identity and a pre-defined embedding"""

    def __init__(self, action_dim=3, n_actions=10, action_cat_dim=10, context_dim=10, config=LearnEmbedConfig()):
        super().__init__()
        self.action_embed_stack = nn.Linear(action_cat_dim * action_dim + n_actions, context_dim)
        self.output = nn.Linear(context_dim, 1)
        nn.init.ones_(self.output.weight)
        self.output.requires_grad_(False)

    def forward(self, context, action, action_embed):
        x1 = self.action_embed_stack(torch.cat([nn.Flatten()(action_embed), action], dim=1))
        out = self.output(torch.mul(x1, context))
        return out

    @disable_training
    def action_embeddings(self, action=None, action_embed=None):
        return self.action_embed_stack(torch.cat([nn.Flatten()(action_embed), action], dim=1))



class LearnEmbedMF(LearnEmbedNetwork):
    """Linear learning objective with low rank MF, where the action network only uses the action identity"""

    def __init__(self, action_dim=3, n_actions=10, action_cat_dim=10, context_dim=10, config=LearnEmbedConfig()):
        super().__init__()
        self.action_embed_stack = nn.Linear(n_actions, config.learned_embed_dim)
        self.context_stack = nn.Linear(context_dim, config.learned_embed_dim)
        self.output = nn.Linear(config.learned_embed_dim, 1)
        nn.init.ones_(self.output.weight)
        self.output.requires_grad_(False)

    def forward(self, context, action, action_embed):
        x1 = self.action_embed_stack(action)
        x2 = self.context_stack(context)
        out = self.output(torch.mul(x1, x2))
        return out

    @disable_training
    def action_embeddings(self, action=None, action_embed=None):
        return self.action_embed_stack(action)



class TorchBanditDataset(Dataset):
    def __init__(self, n_actions, n_dim_action, context, action, action_embed, reward):
        self.context = torch.from_numpy(context).float()
        self.action = torch.nn.functional.one_hot(torch.from_numpy(action), num_classes=n_actions).float()
        self.action_embed = torch.nn.functional.one_hot(torch.from_numpy(action_embed), num_classes=n_dim_action).float()
        self.reward = torch.unsqueeze(torch.from_numpy(reward), 1).float()

    def __len__(self):
        return len(self.reward)

    def __getitem__(self, idx):
        return self.context[idx], self.action[idx], self.action_embed[idx], self.reward[idx]


class LearnedEmbedMIPS(MarginalizedInverseProbabilityWeighting):
    def __init__(
        self,
        embed_model: nn.Module = None,
        learned_embed_params: LearnedEmbedParams = LearnedEmbedParams(),
        embed_model_config: LearnEmbedConfig = LearnEmbedConfig(),
        logging_losses_file=None,
        **kwargs
    ):
        self.embed_model = embed_model
        self.learned_embed_params = learned_embed_params
        self.embed_model_config = embed_model_config
        self.logging_losses_file = logging_losses_file
        super().__init__(**kwargs)

    def _estimate_round_rewards(
        self,
        context: np.ndarray,
        reward: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        action_dist: np.ndarray,
        position: Optional[np.ndarray] = None,
        p_e_a: Optional[np.ndarray] = None,
        with_dev: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_embed: array-like, shape (n_rounds, dim_action_embed)
            Context vectors characterizing actions or action embeddings such as item category information.
            This is used to estimate the marginal importance weights.

        pi_b: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the logging/behavior policy, i.e., :math:`\\pi_b(a_i|x_i)`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, the effect of position on the reward will be ignored.
            (If only a single action is chosen for each data, you can just ignore this argument.)

        p_e_a: array-like, shape (n_actions, n_cat_per_dim, n_cat_dim), default=None
            Conditional distribution of action embeddings given each action.
            This distribution is available only when we use synthetic bandit data, i.e.,
            `obp.dataset.SyntheticBanditDatasetWithActionEmbeds`.
            See the output of the `obtain_batch_bandit_feedback` argument of this class.
            If `p_e_a` is given, MIPW uses the true marginal importance weights based on this distribution.
            The performance of MIPW with the true weights is useful in synthetic experiments of research papers.

        with_dev: bool, default=False.
            Whether to output a deviation bound with the estimated sample-wise rewards.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        n = reward.shape[0]
        learned_embed = self.learn_action_embed(
            context=context,
            action=action,
            action_embed=action_embed,
            reward=reward,
            **self.learned_embed_params.__dict__
        )
        w_x_e = self._estimate_w_x_e(
            action=action,
            action_embed=learned_embed,
            pi_e=action_dist[np.arange(n), :, position],
            pi_b=pi_b[np.arange(n), :, position],
        )
        self.max_w_x_e = w_x_e.max()

        if with_dev:
            r_hat = reward * w_x_e
            cnf = np.sqrt(np.var(r_hat) / (n - 1))
            cnf *= stats.t.ppf(1.0 - (self.delta / 2), n - 1)

            return r_hat.mean(), cnf

        return reward * w_x_e

    def _estimate_w_x_e(
        self,
        action: np.ndarray,
        action_embed: np.ndarray,
        pi_b: np.ndarray,
        pi_e: np.ndarray,
    ) -> np.ndarray:
        """Estimate the marginal importance weights."""
        n = action.shape[0]
        w_x_a = pi_e / pi_b
        w_x_a = np.where(w_x_a < np.inf, w_x_a, 0)
        pi_a_e = np.zeros((n, self.n_actions))
        self.pi_a_x_e_estimator.fit(action_embed, action)
        pi_a_e[:, np.unique(action)] = self.pi_a_x_e_estimator.predict_proba(action_embed)
        w_x_e = (w_x_a * pi_a_e).sum(1)

        return w_x_e

    def learn_action_embed(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_embed: np.ndarray,
        reward: np.ndarray,
        lr=1e-2,
        epochs=250,
        batch_size=32,
        train_test_ratio=0.9,
    ) -> np.ndarray:
        """Learn action embeddings"""
        n_dim_action = len(np.unique(action_embed))
        dataset = TorchBanditDataset(self.n_actions, n_dim_action, context, action, action_embed, reward)
        if train_test_ratio >= 1:
            train_data = dataset
            train_positives = dataset.reward.sum().item()
            train_length = len(dataset)
        else:
            train_data, test_data = random_split(dataset, [round(train_test_ratio * len(dataset)), round((1 - train_test_ratio) * len(dataset))])
            test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
            test_positives = dataset.reward[test_data.indices].sum().item()
            test_length = len(dataset.reward[test_data.indices])
            train_length = len(dataset.reward[train_data.indices])
            train_positives = dataset.reward[train_data.indices].sum().item()

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        model = self.embed_model(n_actions=self.n_actions, action_dim=action_embed.shape[1], action_cat_dim=n_dim_action, context_dim=context.shape[1], config=self.embed_model_config)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        logs = pd.DataFrame()

        for e in range(epochs):
            model.train_loop(train_dataloader, loss_fn, optimizer)
            if self.logging_losses_file and train_test_ratio < 1:
                train_mse = model.report_loss(train_dataloader, loss_fn)
                test_mse = model.report_loss(test_dataloader, loss_fn)
                logs = logs.append(pd.Series(
                    [e, train_mse, test_mse, self.estimator_name, asdict(self.learned_embed_params), asdict(self.embed_model_config), train_positives, train_length, test_positives, test_length],
                    index=["epoch", "train_mse", "test_mse", "model", "learned_embed_params", "embed_model_config", "train_positives", "train_length", "test_positives", "test_length"]
                ), ignore_index=True
                )

        if self.logging_losses_file and train_test_ratio < 1:
            logs.to_parquet(self.logging_losses_file, index=False)

        return model.action_embeddings(dataset.action, dataset.action_embed).detach().numpy()
