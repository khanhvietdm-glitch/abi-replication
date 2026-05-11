"""
Six performance metrics from Section 7.5.

Each metric is computed per epoch and aggregated across the K replications
in `experiments/run_baseline.py`.

Metrics (lower is better unless noted):
    (i)   false_trust_rate          -- high-trust outputs whose true quality < tau
    (ii)  social_welfare            -- net trust value - capital DWL - audit cost - slashing
    (iii) high_type_participation   -- HIGHER better
    (iv)  reward_gini               -- concentration of producer rewards
    (v)   collusion_success_rate    -- coalitions whose realized gain > expected slashing
    (vi)  capital_exclusion_rate    -- high-type producers excluded by stake threshold
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict
import numpy as np

from .agents import ProducerPopulation, ValidatorPopulation
from .config import SimulationParams
from .utils import gini


@dataclass
class EpochMetrics:
    false_trust_rate: float
    social_welfare: float
    high_type_participation: float
    reward_gini: float
    collusion_success_rate: float
    capital_exclusion_rate: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _collusion_success(
    coalition_mask: np.ndarray,
    rewards: np.ndarray,
    slashing: np.ndarray,
) -> float:
    """
    Coalition "succeeds" iff its NET reward share (rewards - slashing) exceeds
    the share it would receive under egalitarian allocation. This matches the
    Section 7.5 definition "realized manipulation gain exceeds the expected
    slashing penalty" while differentiating across mechanisms.
    """
    n = coalition_mask.size
    n_col = int(coalition_mask.sum())
    if n_col == 0 or rewards.sum() <= 0:
        return 0.0
    fair_share = n_col / n
    realized_net = float(rewards[coalition_mask].sum() - slashing[coalition_mask].sum())
    realized_share = realized_net / max(float(rewards.sum()), 1e-12)
    return 1.0 if realized_share > fair_share else 0.0


def compute_metrics(
    prod: ProducerPopulation,
    val: ValidatorPopulation,
    Q_hat: np.ndarray,
    rewards: np.ndarray,
    slashing: np.ndarray,
    audited: np.ndarray,
    eligible: np.ndarray,
    params: SimulationParams,
    epoch_audit_cost: float,
    coalition_realized_gain: float,
    coalition_expected_slash: float,
) -> EpochMetrics:
    """Compute the six headline metrics for one epoch."""
    # --- (i) false-trust rate -----------------------------------------------
    # High-trust = top-quartile reward share; "harmful" = true q < tau.
    tau = params.user_threshold_tau
    if rewards.sum() <= 0:
        false_trust = 0.0
    else:
        threshold = np.quantile(rewards, 0.75)
        trusted = rewards >= threshold
        n_trusted = max(trusted.sum(), 1)
        false_trust = float(np.mean((prod.quality < tau) & trusted) * prod.quality.size / n_trusted)
        false_trust = float(np.clip(false_trust, 0.0, 1.0))

    # --- (ii) social welfare -------------------------------------------------
    # Quality value the user *captures* from trusted outputs minus harms minus costs.
    v_quality = params.welfare_quality_value
    cost_harm = params.welfare_harm_cost
    trust_score_value = float(
        v_quality * np.sum(rewards * prod.quality)
        - cost_harm * np.sum(rewards * np.maximum(0.0, tau - prod.quality))
    )
    capital_dwl = params.capital_deadweight * float(prod.stake.sum())
    welfare = trust_score_value - capital_dwl - epoch_audit_cost - float(slashing.sum())

    # --- (iii) high-type participation --------------------------------------
    high_idx = prod.type_high
    if high_idx.sum() == 0:
        ht_part = 0.0
    else:
        ht_part = float(eligible[high_idx].mean())

    # --- (iv) reward Gini ----------------------------------------------------
    g = gini(rewards)

    # --- (v) collusion success rate -----------------------------------------
    success = _collusion_success(prod.in_coalition, rewards, slashing)

    # --- (vi) capital-exclusion rate ----------------------------------------
    # High-type producers that fail eligibility because of insufficient stake.
    excluded = (~eligible) & high_idx
    if high_idx.sum() == 0:
        cap_excl = 0.0
    else:
        cap_excl = float(excluded.sum()) / float(high_idx.sum())

    return EpochMetrics(
        false_trust_rate=false_trust,
        social_welfare=welfare,
        high_type_participation=ht_part,
        reward_gini=g,
        collusion_success_rate=success,
        capital_exclusion_rate=cap_excl,
    )
