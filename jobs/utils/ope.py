from typing import Dict, Optional

import numpy as np
from obp.ope import DirectMethod as DM
from obp.ope import DoublyRobust as DR
from obp.ope import InverseProbabilityWeighting as IPS
from obp.ope import MarginalizedInverseProbabilityWeighting as MIPS
from obp.ope import OffPolicyEvaluation
from obp.ope import SelfNormalizedInverseProbabilityWeighting as SNIPS
from obp.ope import SwitchDoublyRobustTuning as SwitchDR

from .learn_embed import LearnedEmbedMIPS, LearnEmbedLinear, LearnEmbedLinearB, LearnEmbedLinearC


def run_ope(
    val_bandit_data: Dict,
    action_dist_val: np.ndarray,
    estimated_rewards: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, float]:
    n_actions = val_bandit_data["n_actions"]

    ope_estimators = [
        IPS(estimator_name="IPS"),
        DR(estimator_name="DR"),
        DM(estimator_name="DM"),
        MIPS(n_actions=n_actions, estimator_name="MIPS"),
        MIPS(n_actions=n_actions, estimator_name="MIPS (true)"),
        LearnedEmbedMIPS(estimator_name="Learned MIPS OneHot", n_actions=n_actions, embed_model=LearnEmbedLinear, **kwargs),
        LearnedEmbedMIPS(estimator_name="Learned MIPS FineTune", n_actions=n_actions, embed_model=LearnEmbedLinearB, **kwargs),
        LearnedEmbedMIPS(estimator_name="Learned MIPS Combined", n_actions=n_actions, embed_model=LearnEmbedLinearC, **kwargs),
    ]

    ope = OffPolicyEvaluation(
        bandit_feedback=val_bandit_data,
        ope_estimators=ope_estimators,
    )
    estimated_policy_values = ope.estimate_policy_values(
        action_dist=action_dist_val,
        estimated_rewards_by_reg_model=estimated_rewards,
        action_embed=val_bandit_data["action_embed"],
        pi_b=val_bandit_data["pi_b"],
        p_e_a={"MIPS (true)": val_bandit_data["p_e_a"]},
    )

    return estimated_policy_values


def run_real_dataset_ope(
    val_bandit_data: Dict,
    action_dist_val: np.ndarray,
    estimated_rewards: np.ndarray,
    policy_value: float,
    **kwargs
) -> np.ndarray:
    n_actions = val_bandit_data["n_actions"]
    lambdas = [10, 50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, np.inf]

    ope = OffPolicyEvaluation(
        bandit_feedback=val_bandit_data,
        ope_estimators=[
            IPS(estimator_name="IPS"),
            DR(estimator_name="DR"),
            DM(estimator_name="DM"),
            SNIPS(estimator_name="SNIPS"),
            SwitchDR(lambdas=lambdas, tuning_method="slope", estimator_name="SwitchDR"),
            MIPS(estimator_name="MIPS", n_actions=n_actions),
            MIPS(estimator_name="MIPS (w/ SLOPE)", n_actions=n_actions, embedding_selection_method="exact"),
            LearnedEmbedMIPS(estimator_name="Learned MIPS OneHot", n_actions=n_actions, embed_model=LearnEmbedLinear, **kwargs),
            LearnedEmbedMIPS(estimator_name="Learned MIPS FineTune", n_actions=n_actions, embed_model=LearnEmbedLinearB, **kwargs),
            LearnedEmbedMIPS(estimator_name="Learned MIPS Combined", n_actions=n_actions, embed_model=LearnEmbedLinearC, **kwargs),
        ],
    )

    squared_errors = ope.evaluate_performance_of_estimators(
        ground_truth_policy_value=policy_value,
        action_dist=action_dist_val,
        estimated_rewards_by_reg_model=estimated_rewards,
        action_embed=val_bandit_data["action_context"],
        pi_b=val_bandit_data["pi_b"],
        metric="se",
    )

    relative_squared_errors = {}
    baseline = squared_errors["MIPS (w/ SLOPE)"]
    for key, value in squared_errors.items():
        relative_squared_errors[key] = value / baseline

    return squared_errors, relative_squared_errors
