"""
Shared simulator: builds populations, runs the epoch loop, applies a
mechanism's hooks, and records per-epoch metrics.

References:
    Section 7.5  -- environment + per-epoch steps (a)-(e)
    Appendix C.3 -- random-seed protocol (paired comparison across mechanisms)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

from .agents import (
    ProducerPopulation,
    ValidatorPopulation,
    UserPopulation,
    build_producers,
    build_validators,
    build_users,
    draw_producer_quality,
)
from .config import SimulationParams
from .mechanisms import Mechanism, ABIMechanism
from .metrics import EpochMetrics, compute_metrics


@dataclass
class ReplicationResult:
    mechanism: str
    seed: int
    per_epoch: pd.DataFrame
    final_S_star: float
    final_stake_concentration: float
    final_validator_reliability: np.ndarray


# ---------------------------------------------------------------------------
# Helper for stake-quality elasticity used by ABI governance feedback
# ---------------------------------------------------------------------------
def _stake_quality_elasticity(stake: np.ndarray, quality: np.ndarray) -> float:
    """OLS slope of quality on log(stake)."""
    x = np.log1p(stake)
    if x.std() < 1e-9:
        return 0.0
    return float(np.cov(quality, x, bias=True)[0, 1] / max(x.var(), 1e-12))


# ---------------------------------------------------------------------------
# Helper for speculative contamination ratio Omega
# ---------------------------------------------------------------------------
def _contamination_ratio(committed: np.ndarray, speculative: np.ndarray) -> float:
    var_c = float(np.var(committed))
    var_n = float(np.var(speculative))
    return var_n / max(var_c + var_n, 1e-12)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class Simulator:
    """Single-replication simulator for one mechanism / one master seed."""

    def __init__(
        self,
        params: SimulationParams,
        mechanism: Mechanism,
        master_seed: int,
        total_reward_budget: float = 100.0,
    ):
        self.params = params
        self.mech = mechanism
        self.seed = master_seed
        self.E = total_reward_budget

        # Build populations (paired across mechanisms because seed protocol is shared)
        self.prod = build_producers(params, master_seed)
        self.val = build_validators(params, master_seed)
        self.users = build_users(params, master_seed)

        # Streams used inside the epoch loop
        self.rng_q = np.random.default_rng(master_seed * 7919 + 1)
        self.rng_n = np.random.default_rng(master_seed * 7919 + 2)
        self.rng_s = np.random.default_rng(master_seed * 7919 + 3)
        self.rng_a = np.random.default_rng(master_seed * 7919 + 4)
        self.rng_c = np.random.default_rng(master_seed * 7919 + 5)

    # ------------------------------------------------------------------
    def _validator_reports(self, q: np.ndarray) -> np.ndarray:
        """y_{ji} = q_i + eps_ji  -- shape (N_v, N_p). Coalition members bias up."""
        N_v = self.val.noise_sigma.size
        N_p = q.size
        sigma = self.val.noise_sigma[:, None]
        noise = self.rng_n.normal(0.0, 1.0, size=(N_v, N_p)) * sigma
        y = q[None, :] + noise
        # Coalition bias: validators in coalition push up reports for colluding producers.
        if self.val.in_coalition.any():
            bias = np.zeros_like(y)
            colluder_prod = self.prod.in_coalition
            colluder_val = self.val.in_coalition
            # Push targeted producer reports up by 0.3 (large enough to move ranking).
            bias[np.ix_(colluder_val, colluder_prod)] = 0.30
            y = y + bias
        return np.clip(y, 0.0, 1.0)

    # ------------------------------------------------------------------
    def _audit_outcome(self, audited: np.ndarray) -> np.ndarray:
        """audit_fail ~ Bernoulli(pi_H) for high-type, Bernoulli(pi_L) for low-type."""
        p = self.params
        pi = np.where(self.prod.type_high, p.audit_failure_prob_H, p.audit_failure_prob_L)
        u = self.rng_a.random(size=self.prod.stake.size)
        return audited & (u < pi)

    # ------------------------------------------------------------------
    def _update_speculative_stake(self, t: int) -> None:
        """Add/remove speculative flow each epoch (contaminates stake signal)."""
        p = self.params
        N = self.prod.stake.size
        # Speculative flow scales with intensity parameter and is mean-zero by design.
        flow = self.rng_s.normal(0.0, 1.0, size=N) * p.phi_speculative
        # Speculative stake stays non-negative for accounting purposes.
        self.prod.speculative_stake = np.clip(
            0.8 * self.prod.speculative_stake + flow, 0.0, None
        )

    # ------------------------------------------------------------------
    def _producer_stake_choice(self) -> None:
        """
        Stake decision rule (calibrated, not strategic):
          - high-type producers stake aggressively toward S* (subject to capital)
          - low-type producers stake just enough to clear the eligibility test
          - speculative stake is added on top per Theorem 5
        """
        cap = self.prod.capital_endow
        target = np.where(self.prod.type_high, self.mech.S_star, 0.45 * self.mech.S_star)
        # Capital constraint: cannot post more than endowment.
        commit = np.minimum(target, cap)
        self.prod.stake = commit + self.prod.speculative_stake

    # ------------------------------------------------------------------
    def _coalition_gain_and_slash(
        self,
        audited: np.ndarray,
        audit_fail: np.ndarray,
        slashing: np.ndarray,
    ) -> tuple:
        """Aggregate manipulation gain and expected slashing for the coalition."""
        col_prod_mask = self.prod.in_coalition
        # Manipulation gain realized this epoch: G_i for each colluding producer.
        realized_gain = float(self.prod.manipulation_gain[col_prod_mask].sum())
        # Expected slashing on the coalition's stake (Theorem 12, Eq. 197).
        d = self.params.audit_detect_d
        phi = self.params.slash_rate_purestake
        A_i = self.params.audit_capacity_frac
        per_producer = A_i * d * phi * self.prod.stake[col_prod_mask]
        per_validator = A_i * d * phi * self.val.stake[self.val.in_coalition]
        expected_slash = float(per_producer.sum() + per_validator.sum())
        return realized_gain, expected_slash

    # ------------------------------------------------------------------
    def run(self) -> ReplicationResult:
        T = self.params.T
        rows: List[dict] = []

        # Mark colluding producers: a random subset on the producer side
        # also acts as accomplices to colluding validators.
        if self.val.in_coalition.any():
            n_col_prod = max(1, int(round(self.prod.stake.size * 0.05)))
            idx = self.rng_c.choice(self.prod.stake.size, size=n_col_prod, replace=False)
            self.prod.in_coalition[idx] = True

        for t in range(T):
            # (0) Speculative flow update (Section 6.2)
            self._update_speculative_stake(t)

            # (1) Producer stake choice + threshold check
            self._producer_stake_choice()
            eligible = self.mech.apply_threshold(self.prod)

            # (2) Validator reports
            draw_producer_quality(self.prod, self.params, self.rng_q)
            y = self._validator_reports(self.prod.quality)
            Q_hat = self.mech.aggregate_reports(y, self.val)

            # (3) Audit selection
            audited = self.mech.select_audit(self.prod, Q_hat, eligible, self.rng_a)
            audit_fail = self._audit_outcome(audited)

            # (4) Reward + slashing
            rewards = self.mech.allocate_rewards(self.prod, Q_hat, eligible, self.E)
            slashing = self.mech.compute_slashing(self.prod, audited, audit_fail, Q_hat)

            # (5) Update reputation, audit score, risk, validator sigma_hat
            #   Reputation: smoothed validator estimate of quality
            alpha_rep = self.params.rep_smoothing
            self.prod.reputation = alpha_rep * self.prod.reputation + (1 - alpha_rep) * Q_hat
            self.prod.reputation = np.clip(self.prod.reputation, 0.0, 1.0)
            # Audit score: smoothed pass rate
            self.prod.audit_score = np.where(
                audited,
                0.7 * self.prod.audit_score + 0.3 * (1.0 - audit_fail.astype(float)),
                self.prod.audit_score,
            )
            # Risk score: cumulative anomaly indicator (variance of reports / coalition flag)
            report_var = y.var(axis=0)
            self.prod.risk_score = 0.9 * self.prod.risk_score + 0.1 * (report_var > 0.05).astype(float)

            # Validator sigma_hat update: use audited producers as ground-truth
            if audited.any():
                truth = self.prod.quality[audited]
                err = y[:, audited] - truth[None, :]
                var = (err ** 2).mean(axis=1)
                self.val.sigma_hat_sq = 0.7 * self.val.sigma_hat_sq + 0.3 * var
                # Penalize coalition members faster (collusion-aware update)
                self.val.sigma_hat_sq = np.where(
                    self.val.in_coalition,
                    1.5 * self.val.sigma_hat_sq,
                    self.val.sigma_hat_sq,
                )

            # Apply slashing to stake (cannot go below zero)
            self.prod.stake = np.maximum(0.0, self.prod.stake - slashing)
            # Add rewards back to capital (becomes part of next epoch endowment)
            self.prod.capital_endow = self.prod.capital_endow + 0.1 * rewards

            # (6) Quality credit (ABI only)
            if isinstance(self.mech, ABIMechanism):
                self.mech.update_quality_credit(self.prod, Q_hat)

            # (7) Governance update (only ABI with dynamic threshold)
            p_t = float(eligible.mean())
            beta_t = _stake_quality_elasticity(self.prod.stake, self.prod.quality)
            omega_t = _contamination_ratio(
                self.prod.stake - self.prod.speculative_stake,
                self.prod.speculative_stake,
            )
            self.mech.governance_update(p_t, beta_t, omega_t)

            # (8) Compute metrics for this epoch
            audit_cost = audited.sum() * self.params.kappa_A * self.E / max(self.prod.stake.size, 1)
            real_gain, exp_slash = self._coalition_gain_and_slash(audited, audit_fail, slashing)
            m = compute_metrics(
                prod=self.prod,
                val=self.val,
                Q_hat=Q_hat,
                rewards=rewards,
                slashing=slashing,
                audited=audited,
                eligible=eligible,
                params=self.params,
                epoch_audit_cost=audit_cost,
                coalition_realized_gain=real_gain,
                coalition_expected_slash=exp_slash,
            )
            row = {"epoch": t, "S_star": self.mech.S_star,
                   "p_t": p_t, "beta_t": beta_t, "omega_t": omega_t,
                   **m.as_dict()}
            rows.append(row)

        df = pd.DataFrame(rows)
        return ReplicationResult(
            mechanism=self.mech.name,
            seed=self.seed,
            per_epoch=df,
            final_S_star=self.mech.S_star,
            final_stake_concentration=float(np.std(self.prod.stake) / max(np.mean(self.prod.stake), 1e-9)),
            final_validator_reliability=self.val.sigma_hat_sq.copy(),
        )
