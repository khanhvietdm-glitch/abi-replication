"""
Four trust mechanisms (Section 7.5 + Appendix C.1).

Each mechanism shares the same outer epoch loop (see ``Simulator``) and
overrides four steps:
    * threshold check          -> ``apply_threshold``
    * validator aggregation    -> ``aggregate_reports``
    * audit selection          -> ``select_audit``
    * reward / slashing rules  -> ``allocate_rewards`` + ``compute_slashing``
    * governance update        -> ``governance_update`` (no-op for baselines)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from .agents import ProducerPopulation, ValidatorPopulation
from .config import SimulationParams
from .utils import (
    softmax_rewards,
    inverse_variance_weights,
    stake_transform,
    quality_credit,
    update_log_threshold,
)


# ===========================================================================
# Base interface
# ===========================================================================
class Mechanism(ABC):
    """Abstract base for the four mechanisms compared in Section 7.5."""

    name: str = "base"

    def __init__(self, params: SimulationParams):
        self.params = params
        self.log_S_star = float(np.log(params.S_star_init))

    @property
    def S_star(self) -> float:
        return float(np.exp(self.log_S_star))

    # ----------------- mechanism-specific hooks ----------------------------
    @abstractmethod
    def apply_threshold(self, prod: ProducerPopulation) -> np.ndarray:
        """Return boolean mask of producers eligible for reward this epoch."""

    @abstractmethod
    def aggregate_reports(
        self,
        y: np.ndarray,                  # (N_v, N_p)
        val: ValidatorPopulation,
    ) -> np.ndarray:
        """Return aggregate quality estimate per producer, shape (N_p,)."""

    @abstractmethod
    def select_audit(
        self,
        prod: ProducerPopulation,
        Q_hat: np.ndarray,
        eligible: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return boolean mask of producers to audit this epoch."""

    @abstractmethod
    def allocate_rewards(
        self,
        prod: ProducerPopulation,
        Q_hat: np.ndarray,
        eligible: np.ndarray,
        E_t: float,
    ) -> np.ndarray:
        """Return per-producer reward R_i (shape N_p)."""

    @abstractmethod
    def compute_slashing(
        self,
        prod: ProducerPopulation,
        audited: np.ndarray,
        audit_fail: np.ndarray,
        Q_hat: np.ndarray,
    ) -> np.ndarray:
        """Return per-producer slashing amount Slash_i (shape N_p)."""

    def governance_update(self, p_t: float, beta_t: float, omega_t: float) -> None:
        """No-op for baselines; ABI overrides."""
        return None


# ===========================================================================
# Algorithm C.1 -- Pure-Stake Baseline
# ===========================================================================
class PureStakeMechanism(Mechanism):
    """Rank by stake; linear allocation; flat slashing on audit failure."""

    name = "pure_stake"

    def apply_threshold(self, prod):
        return prod.stake >= self.S_star     # generic ex-ante threshold

    def aggregate_reports(self, y, val):
        # Pure stake ignores quality reports for ranking, but we still expose
        # a coarse estimator for metric computation: equal-weighted mean.
        return y.mean(axis=0)

    def select_audit(self, prod, Q_hat, eligible, rng):
        # Uniform random audit (no risk targeting)
        N = prod.stake.size
        n_audit = int(round(N * self.params.audit_capacity_frac))
        idx = rng.choice(N, size=n_audit, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[idx] = True
        return mask

    def allocate_rewards(self, prod, Q_hat, eligible, E_t):
        weights = np.where(eligible, prod.stake, 0.0)
        total = weights.sum()
        if total <= 0:
            return np.zeros_like(weights)
        return E_t * weights / total

    def compute_slashing(self, prod, audited, audit_fail, Q_hat):
        rate = self.params.slash_rate_purestake
        slashed = np.where(audited & audit_fail, rate * prod.stake, 0.0)
        return slashed


# ===========================================================================
# Algorithm C.2 -- Pure-Reputation Baseline
# ===========================================================================
class PureReputationMechanism(Mechanism):
    """Equal-weighted consensus; reputation smoothing; uniform audit; no slashing."""

    name = "pure_reputation"

    def apply_threshold(self, prod):
        # No stake threshold; everyone is eligible.
        return np.ones_like(prod.stake, dtype=bool)

    def aggregate_reports(self, y, val):
        return y.mean(axis=0)

    def select_audit(self, prod, Q_hat, eligible, rng):
        N = prod.stake.size
        n_audit = int(round(N * self.params.audit_capacity_frac))
        idx = rng.choice(N, size=n_audit, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[idx] = True
        return mask

    def allocate_rewards(self, prod, Q_hat, eligible, E_t):
        # Update of producer reputation is handled in the Simulator;
        # here we allocate proportional to *current* reputation.
        weights = np.where(eligible, prod.reputation, 0.0)
        if weights.sum() <= 0:
            return np.zeros_like(weights)
        return E_t * weights / weights.sum()

    def compute_slashing(self, prod, audited, audit_fail, Q_hat):
        return np.zeros_like(prod.stake)


# ===========================================================================
# Algorithm C.3 -- Yuma-like Baseline (stylized)
# ===========================================================================
class YumaLikeMechanism(Mechanism):
    """Stake-weighted validator aggregation, top-quantile clipping, S * y ranking."""

    name = "yuma_like"

    def apply_threshold(self, prod):
        return prod.stake >= self.S_star

    def aggregate_reports(self, y, val):
        # Stake-weighted aggregation: ŷ_i = sum_j (S_j * y_ji) / sum_j S_j
        w = val.stake / val.stake.sum()
        y_hat = w @ y
        # Clip at top quantile (Yuma-style over-evaluation cap).
        cap = np.quantile(y_hat, self.params.yuma_clip_q)
        return np.minimum(y_hat, cap)

    def select_audit(self, prod, Q_hat, eligible, rng):
        # "Limited" audit -- random over eligible producers only.
        N = prod.stake.size
        n_audit = int(round(N * self.params.audit_capacity_frac))
        eligible_idx = np.flatnonzero(eligible)
        if eligible_idx.size == 0:
            return np.zeros(N, dtype=bool)
        n_audit = min(n_audit, eligible_idx.size)
        chosen = rng.choice(eligible_idx, size=n_audit, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[chosen] = True
        return mask

    def allocate_rewards(self, prod, Q_hat, eligible, E_t):
        score = prod.stake * Q_hat       # stake x clipped score
        weights = np.where(eligible, score, 0.0)
        if weights.sum() <= 0:
            return np.zeros_like(weights)
        return E_t * weights / weights.sum()

    def compute_slashing(self, prod, audited, audit_fail, Q_hat):
        # Generic slashing on audit failure -- same rule as Pure-Stake.
        rate = self.params.slash_rate_purestake
        return np.where(audited & audit_fail, rate * prod.stake, 0.0)


# ===========================================================================
# Algorithm C.4 / Algorithm 1 -- ABI mechanism
# ===========================================================================
class ABIMechanism(Mechanism):
    """
    Full ABI mechanism (Section 8.1, Algorithm 1).

    Substitutions relative to baselines (Appendix C.1):
        2'. Inverse-variance reliability weighting (Proposition 1)
        3'. Risk-targeted audit (Theorem 3)
        4'. Dynamic threshold S* (Theorem 10)
        5'. Quality-conditional capital relief (Proposition 2)

    Flags allow ablation studies (Appendix C.4).
    """

    name = "abi"

    def __init__(
        self,
        params: SimulationParams,
        use_inverse_variance: bool = True,
        use_risk_targeted_audit: bool = True,
        use_dynamic_threshold: bool = True,
        use_quality_credit: bool = True,
    ):
        super().__init__(params)
        self.use_inverse_variance = use_inverse_variance
        self.use_risk_targeted_audit = use_risk_targeted_audit
        self.use_dynamic_threshold = use_dynamic_threshold
        self.use_quality_credit = use_quality_credit

    # ------------------------------------------------------------------
    def effective_stake(self, prod: ProducerPopulation) -> np.ndarray:
        """tilde s_i = s_i + f_i (Eq. 230)."""
        if self.use_quality_credit:
            return prod.stake + prod.quality_credit
        return prod.stake

    def apply_threshold(self, prod: ProducerPopulation) -> np.ndarray:
        return self.effective_stake(prod) >= self.S_star

    # ------------------------------------------------------------------
    def aggregate_reports(self, y: np.ndarray, val: ValidatorPopulation) -> np.ndarray:
        if self.use_inverse_variance:
            w = inverse_variance_weights(val.sigma_hat_sq)
        else:
            # ABI-NoReliability ablation: stake-weighted
            w = val.stake / val.stake.sum()
        return w @ y

    # ------------------------------------------------------------------
    def select_audit(self, prod, Q_hat, eligible, rng):
        N = prod.stake.size
        n_audit = int(round(N * self.params.audit_capacity_frac))

        if not self.use_risk_targeted_audit:
            # ABI-NoAudit ablation: uniform random
            idx = rng.choice(N, size=n_audit, replace=False)
            mask = np.zeros(N, dtype=bool)
            mask[idx] = True
            return mask

        # Risk-targeted: minimum audit probability A_i^min = G_i / (d * phi * s)
        # (Theorem 3). Audit producers with the largest required A_i^min.
        d = self.params.audit_detect_d
        phi = self.params.slash_rate_purestake
        s_eff = self.effective_stake(prod)
        A_min = prod.manipulation_gain / np.maximum(d * phi * s_eff, 1e-6)
        # Combine with a baseline-anomaly score that flags low Q_hat producers.
        priority = A_min + self.params.target_audit_gain_share * (1.0 - Q_hat)
        priority = np.where(eligible, priority, -np.inf)
        order = np.argsort(-priority)
        mask = np.zeros(N, dtype=bool)
        mask[order[:n_audit]] = True
        return mask

    # ------------------------------------------------------------------
    def composite_trust_score(self, prod: ProducerPopulation, Q_hat: np.ndarray) -> np.ndarray:
        """z_i = lambda_Q Q + lambda_S g(s_eff/S*) + lambda_R Rep + lambda_A Audit - lambda_X Risk."""
        p = self.params
        s_eff = self.effective_stake(prod)
        stake_term = stake_transform(s_eff / max(self.S_star, 1e-6))
        # Re-use ABI calibration weights from config baseline.
        from .config import ABI_BASELINE as ABI
        z = (
            ABI.lambda_Q * Q_hat
            + ABI.lambda_S * stake_term
            + ABI.lambda_R * prod.reputation
            + ABI.lambda_A * prod.audit_score
            - ABI.lambda_X * prod.risk_score
        )
        return z

    def allocate_rewards(self, prod, Q_hat, eligible, E_t):
        z = self.composite_trust_score(prod, Q_hat)
        z = np.where(eligible, z, -1e6)         # exclude ineligible
        rewards = softmax_rewards(z, self.params.eta_softmax, E_t)
        return np.where(eligible, rewards, 0.0)

    # ------------------------------------------------------------------
    def compute_slashing(self, prod, audited, audit_fail, Q_hat):
        """Quality-quadratic slashing: Slash_i = phi * s_i * max(0, tau - Q_i)^2  (Eq. 233)."""
        phi = self.params.slash_rate_purestake
        tau = self.params.Q_min
        deficit = np.maximum(0.0, tau - Q_hat)
        slashed = phi * prod.stake * (deficit ** 2)
        # Only realized on actual audit failure
        return np.where(audited & audit_fail, slashed, 0.0)

    # ------------------------------------------------------------------
    def update_quality_credit(self, prod: ProducerPopulation, Q_hat: np.ndarray) -> None:
        """Proposition 2: f_i = min(f_max, psi * max(0, Q - Q_min) * Rep)."""
        if not self.use_quality_credit:
            prod.quality_credit = np.zeros_like(prod.stake)
            return
        f_max = self.params.f_max_credit * self.S_star
        new = quality_credit(
            Q=Q_hat,
            Rep=prod.reputation,
            Q_min=self.params.Q_min,
            psi=self.params.psi_credit,
            f_max=f_max,
        )
        # Clawback rule (Eq. 172): if Q drops, attenuate previous credit
        decay = np.where(Q_hat < self.params.Q_min, 1.0 - self.params.delta_clawback, 1.0)
        prod.quality_credit = np.maximum(new, decay * prod.quality_credit)

    # ------------------------------------------------------------------
    def governance_update(self, p_t: float, beta_t: float, omega_t: float) -> None:
        if not self.use_dynamic_threshold:
            return
        p = self.params
        # Use small gains kappa, xi, zeta from ABI calibration.
        from .config import ABI_BASELINE as ABI
        kappa = ABI.kappa_g * p.governance_gain
        xi = ABI.xi * p.governance_gain
        zeta = ABI.zeta * p.governance_gain
        self.log_S_star = update_log_threshold(
            self.log_S_star,
            p_t=p_t,
            p_star=p.p_star,
            beta_t=beta_t,
            beta_star=p.beta_star,
            omega_t=omega_t,
            kappa=kappa,
            xi=xi,
            zeta=zeta,
        )
        # Mild guardrails: prevent log-threshold drift far from feasible region.
        self.log_S_star = float(np.clip(self.log_S_star, -3.0, 3.0))


# ===========================================================================
# Factory
# ===========================================================================
def build_mechanism(name: str, params: SimulationParams, **kwargs) -> Mechanism:
    """Construct a mechanism by name. Used by CLI scripts."""
    table = {
        "pure_stake": PureStakeMechanism,
        "pure_reputation": PureReputationMechanism,
        "yuma_like": YumaLikeMechanism,
        "abi": ABIMechanism,
        # Ablation variants
        "abi_no_audit": lambda p: ABIMechanism(p, use_risk_targeted_audit=False),
        "abi_no_credit": lambda p: ABIMechanism(p, use_quality_credit=False),
        "abi_no_dynamic": lambda p: ABIMechanism(p, use_dynamic_threshold=False),
        "abi_no_reliability": lambda p: ABIMechanism(p, use_inverse_variance=False),
    }
    if name not in table:
        raise ValueError(f"Unknown mechanism: {name}; expected one of {list(table)}")
    entry = table[name]
    if callable(entry) and not isinstance(entry, type):
        return entry(params)
    return entry(params, **kwargs)
