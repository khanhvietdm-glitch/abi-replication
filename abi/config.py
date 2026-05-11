"""
Baseline parameter sets.

Numeric values reproduce Table 6 (Section 7.1, illustrative ABI calibration)
and Table C1 (Appendix C.2, agent-based simulation parameters).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple


@dataclass
class Params:
    """Generic parameter container with dict-like serialization."""

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ABICalibration(Params):
    """Table 6: illustrative ABI calibration."""

    pi_H: float = 0.08          # audit-failure prob, high type
    pi_L: float = 0.32          # audit-failure prob, low type
    m: float = 0.045            # marginal reward of stake
    r_f: float = 0.020          # risk-free opportunity cost
    phi: float = 0.130          # slashing intensity (interior of feasible interval)
    d: float = 0.850            # audit detection probability
    lambda_Q: float = 0.55      # quality weight
    lambda_S: float = 0.20      # stake weight
    lambda_R: float = 0.10      # reputation weight
    lambda_A: float = 0.15      # audit weight
    lambda_X: float = 0.10      # risk penalty weight
    eta: float = 3.0            # softmax temperature
    kappa_g: float = 0.10       # threshold gain on participation
    xi: float = 0.20            # threshold gain on elasticity
    zeta: float = 0.30          # threshold gain on contamination
    f_max_frac: float = 0.35    # f_max = 0.35 * S*


@dataclass
class SimulationParams(Params):
    """Table C1: agent-based simulation parameters."""

    N_p: int = 200              # producers
    N_v: int = 40               # validators
    N_u: int = 5_000            # users
    T: int = 200                # epochs per replication
    burn_in: int = 50           # initial epochs ignored when aggregating
    K: int = 50                 # replications

    high_type_share: float = 0.40
    alpha_H: float = 6.0
    beta_H: float = 2.0
    alpha_L: float = 2.0
    beta_L: float = 5.0

    sigma_v: float = 0.15       # validator noise std (mean honest)
    sigma_v_collude: float = 0.30  # additional noise for honest reporters when colluders bias

    phi_speculative: float = 0.20   # speculative-flow intensity
    pi_c: float = 0.10              # coalition fraction
    kappa_A: float = 0.05           # audit cost per unit reward emission

    # Mechanism-specific:
    slash_rate_purestake: float = 0.30
    rep_smoothing: float = 0.85
    yuma_clip_q: float = 0.95
    eta_softmax: float = 5.0
    S_star_init: float = 1.0
    governance_gain: float = 0.40

    # Audit infrastructure:
    audit_capacity_frac: float = 0.10   # fraction of producers audited each epoch
    audit_detect_d: float = 0.85
    audit_failure_prob_H: float = 0.08
    audit_failure_prob_L: float = 0.32

    # Quality credit (Proposition 2):
    Q_min: float = 0.5
    psi_credit: float = 1.5
    f_max_credit: float = 0.35      # f_max = 0.35 * S*
    delta_clawback: float = 0.5     # credit decay when quality drops below Q_min

    # User loss tolerance:
    user_alpha: float = 2.0
    user_beta: float = 5.0
    user_threshold_tau: float = 0.5   # acceptance threshold q >= tau

    # Welfare weights:
    welfare_quality_value: float = 1.0   # value of one unit of trusted quality
    welfare_harm_cost: float = 3.0       # cost of one unit of false-trust
    capital_deadweight: float = 0.02     # cost of locking capital (r_f)

    # Risk-targeted audit (Theorem 3):
    target_audit_gain_share: float = 0.30   # how aggressively to target high-G producers

    # Reliability update (Step 1 of Algorithm 1):
    rho_eta: float = 1.5         # softmax temperature in reliability rule

    # Dynamic threshold (Theorem 10) reference values:
    p_star: float = 0.50
    beta_star: float = 0.30

    seed_base: int = 1


# --- Concrete baseline instances -------------------------------------------
ABI_BASELINE: ABICalibration = ABICalibration()
SIM_BASELINE: SimulationParams = SimulationParams()


# Sensitivity grids from Appendix C.5
SWEEP_PHI: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
SWEEP_PI_C: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
SWEEP_KAPPA_A: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
