"""
Agent populations: producers, validators, users.

References:
    Section 3.1 -- agents and information structure
    Appendix C  -- baseline simulation populations

Population state is held in plain numpy arrays. This keeps the inner
loop fast and easy to inspect. Each agent type is a small dataclass
returning these arrays via a factory; no per-agent objects are created.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimulationParams
from .utils import make_rng


# ---------------------------------------------------------------------------
# Producers
# ---------------------------------------------------------------------------
@dataclass
class ProducerPopulation:
    """Per-producer arrays. Indexed by i in [0, N_p)."""
    type_high: np.ndarray         # bool, True for high-type
    quality: np.ndarray           # latent quality q_i in [0,1] for current epoch
    stake: np.ndarray             # posted stake s_i >= 0
    reputation: np.ndarray        # Rep_i in [0,1]
    audit_score: np.ndarray       # Audit_i (rolling pass rate)
    risk_score: np.ndarray        # Risk_i (anomaly indicator)
    capital_endow: np.ndarray     # initial capital budget
    speculative_stake: np.ndarray # contamination component N_i (Theorem 5)
    quality_credit: np.ndarray    # f_i for ABI (Proposition 2)
    in_coalition: np.ndarray      # bool; collusion only matters for validators here
    manipulation_gain: np.ndarray # G_i, expected per-epoch private gain from manipulation


def build_producers(params: SimulationParams, master_seed: int) -> ProducerPopulation:
    rng_q = make_rng(master_seed, "quality")
    rng_a = make_rng(master_seed, "audit_selection")
    N = params.N_p
    n_high = int(round(N * params.high_type_share))
    type_high = np.zeros(N, dtype=bool)
    type_high[:n_high] = True
    rng_a.shuffle(type_high)

    # Capital endowments: heavy-tailed. High-type producers are *not* on average richer;
    # this is essential to make Proposition 2 testable.
    capital = rng_q.lognormal(mean=0.0, sigma=1.0, size=N)
    capital = capital / capital.mean()    # normalize: mean endowment = 1
    # Initial stake = some fraction of capital, clipped at threshold for first epoch.
    init_stake = 0.5 * capital

    rep_init = np.where(type_high, 0.6, 0.4)

    # Manipulation gain G_i: low-type producers gain more from passing as high-type.
    G = np.where(type_high, 0.05, 0.20)

    return ProducerPopulation(
        type_high=type_high,
        quality=np.zeros(N),
        stake=init_stake,
        reputation=rep_init,
        audit_score=np.full(N, 0.5),
        risk_score=np.zeros(N),
        capital_endow=capital,
        speculative_stake=np.zeros(N),
        quality_credit=np.zeros(N),
        in_coalition=np.zeros(N, dtype=bool),
        manipulation_gain=G,
    )


def draw_producer_quality(pop: ProducerPopulation, params: SimulationParams, rng: np.random.Generator) -> None:
    """Refresh latent q_i from Beta(alpha, beta) per type."""
    N = pop.type_high.size
    q = np.empty(N)
    high_idx = pop.type_high
    q[high_idx] = rng.beta(params.alpha_H, params.beta_H, size=high_idx.sum())
    q[~high_idx] = rng.beta(params.alpha_L, params.beta_L, size=(~high_idx).sum())
    pop.quality = q


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
@dataclass
class ValidatorPopulation:
    """Per-validator arrays. Indexed by j in [0, N_v)."""
    noise_sigma: np.ndarray       # baseline std of validator j's error
    reliability: np.ndarray       # rho_j, updated by audit deviations
    stake: np.ndarray             # validator b_j (used by Yuma stake-weighted aggregation)
    in_coalition: np.ndarray      # bool; coalition members coordinate biased reports
    sigma_hat_sq: np.ndarray      # audit-estimated noise variance


def build_validators(params: SimulationParams, master_seed: int) -> ValidatorPopulation:
    rng_n = make_rng(master_seed, "validator_noise")
    rng_c = make_rng(master_seed, "coalition_formation")
    N = params.N_v

    # Heterogeneous baseline noise sigma_j. Median around params.sigma_v.
    sigma = rng_n.lognormal(mean=np.log(params.sigma_v), sigma=0.4, size=N)

    # Validator stake: heavy-tailed. Required by Yuma-like aggregation.
    stake = rng_n.lognormal(mean=0.0, sigma=1.0, size=N)
    stake = stake / stake.mean()

    # Coalition members
    n_col = int(round(N * params.pi_c))
    in_col = np.zeros(N, dtype=bool)
    if n_col > 0:
        idx = rng_c.choice(N, size=n_col, replace=False)
        in_col[idx] = True

    return ValidatorPopulation(
        noise_sigma=sigma,
        reliability=np.full(N, 1.0 / N),
        stake=stake,
        in_coalition=in_col,
        sigma_hat_sq=np.full(N, params.sigma_v ** 2),
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
@dataclass
class UserPopulation:
    """User loss tolerance for the welfare metric."""
    loss_tolerance: np.ndarray


def build_users(params: SimulationParams, master_seed: int) -> UserPopulation:
    rng = make_rng(master_seed, "quality")
    tol = rng.beta(params.user_alpha, params.user_beta, size=params.N_u)
    return UserPopulation(loss_tolerance=tol)
