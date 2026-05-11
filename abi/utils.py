"""
Mathematical helpers shared across mechanisms.

References:
    Lemma 1                -- softmax reward gradient
    Proposition 1          -- inverse-variance validator aggregation
    Theorem 6              -- saturating stake transformation g(x) = log(1+x)
    Theorem 10 / Eq. (228) -- dynamic stake threshold recursion
"""
from __future__ import annotations

import hashlib
import numpy as np
from typing import Iterable


# ---------------------------------------------------------------------------
# Softmax allocation (Lemma 1, Eq. 141)
# ---------------------------------------------------------------------------
def softmax_rewards(z: np.ndarray, eta: float, total_budget: float) -> np.ndarray:
    """Reward law R_i = E * softmax_eta(z)_i (Eq. 141)."""
    z = np.asarray(z, dtype=float)
    # Numerical-stability shift
    shifted = eta * (z - z.max())
    exps = np.exp(shifted)
    return total_budget * exps / exps.sum()


# ---------------------------------------------------------------------------
# Inverse-variance aggregation (Proposition 1, Eq. 156)
# ---------------------------------------------------------------------------
def inverse_variance_weights(sigma2: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """w_j ∝ 1/sigma_j^2, normalized to sum to one."""
    sigma2 = np.asarray(sigma2, dtype=float)
    inv = 1.0 / np.maximum(sigma2, eps)
    return inv / inv.sum()


def inverse_variance_estimate(y: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """
    Per-producer estimator hat{Q}_i = sum_j w_j y_{ij}  (Eq. 154).

    Parameters
    ----------
    y : array (N_v, N_p) of validator scores
    sigma2 : array (N_v,) per-validator noise variances
    """
    w = inverse_variance_weights(sigma2)
    return w @ y


# ---------------------------------------------------------------------------
# Stake transformation (Theorem 6, baseline g(x) = log(1+x))
# ---------------------------------------------------------------------------
def stake_transform(x: np.ndarray) -> np.ndarray:
    """Concave, marginally saturating g(x) = log(1+x)."""
    return np.log1p(np.maximum(x, 0.0))


# ---------------------------------------------------------------------------
# Dynamic stake threshold update (Theorem 10, Eq. 228)
# ---------------------------------------------------------------------------
def update_log_threshold(
    log_S: float,
    p_t: float,
    p_star: float,
    beta_t: float,
    beta_star: float,
    omega_t: float,
    kappa: float,
    xi: float,
    zeta: float,
) -> float:
    """log S*_{t+1} = log S*_t + kappa(p - p*) + xi(beta* - beta) + zeta * Omega."""
    return log_S + kappa * (p_t - p_star) + xi * (beta_star - beta_t) + zeta * omega_t


# ---------------------------------------------------------------------------
# Concentration metric (Reward Gini, used in Section 7.5)
# ---------------------------------------------------------------------------
def gini(values: np.ndarray, eps: float = 1e-12) -> float:
    """Gini coefficient of a non-negative vector."""
    values = np.asarray(values, dtype=float).flatten()
    if values.size == 0:
        return 0.0
    values = np.clip(values, 0.0, None)
    if values.sum() < eps:
        return 0.0
    sorted_v = np.sort(values)
    n = sorted_v.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * sorted_v).sum() - (n + 1) * sorted_v.sum()) / (n * sorted_v.sum()))


# ---------------------------------------------------------------------------
# Seed protocol from Appendix C.3
# ---------------------------------------------------------------------------
def derive_seeds(master_seed: int) -> dict:
    """Derive five sub-stream seeds from a master seed via SHA-256."""
    salts = {
        "quality": 0x01,
        "validator_noise": 0x02,
        "speculative_flow": 0x03,
        "audit_selection": 0x04,
        "coalition_formation": 0x05,
    }
    out = {}
    for name, salt in salts.items():
        h = hashlib.sha256(f"{master_seed}-{salt}".encode()).digest()
        out[name] = int.from_bytes(h[:4], byteorder="big")
    return out


def make_rng(master_seed: int, stream: str) -> np.random.Generator:
    """Return a numpy Generator for a given (seed, stream) pair."""
    seeds = derive_seeds(master_seed)
    if stream not in seeds:
        raise KeyError(f"Unknown stream {stream!r}; expected one of {list(seeds)}")
    return np.random.default_rng(seeds[stream])


# ---------------------------------------------------------------------------
# Binary entropy h_2 (Proposition 3 / Theorem 4)
# ---------------------------------------------------------------------------
def binary_entropy(p: float) -> float:
    """Binary entropy in nats. h_2(0) = h_2(1) = 0."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


# ---------------------------------------------------------------------------
# Speculative contamination decomposition (Theorem 5)
# ---------------------------------------------------------------------------
def attenuated_elasticity(beta_commit: float, omega: float) -> float:
    """beta_observed = (1 - Omega) * beta_commit (Eq. for Theorem 5)."""
    return (1.0 - omega) * beta_commit


def quality_credit(
    Q: np.ndarray,
    Rep: np.ndarray,
    Q_min: float,
    psi: float,
    f_max: float,
) -> np.ndarray:
    """f_i = min(f_max, psi * max(0, Q_i - Q_min) * Rep_i)  (Eq. 169)."""
    raw = psi * np.maximum(0.0, Q - Q_min) * Rep
    return np.minimum(f_max, raw)


def feasible_slashing_interval(m: float, r_f: float, pi_H: float, pi_L: float) -> tuple:
    """Return (phi_min, phi_max) from Theorem 2: [(m - r_f)/pi_L, (m - r_f)/pi_H]."""
    return (m - r_f) / pi_L, (m - r_f) / pi_H


def feasible_marginal_reward_band(phi: float, r_f: float, pi_H: float, pi_L: float) -> tuple:
    """Theorem 1: r_f + pi_H * phi <= m <= r_f + pi_L * phi."""
    return r_f + pi_H * phi, r_f + pi_L * phi


def min_audit_probability(G: np.ndarray, d: np.ndarray, phi: np.ndarray, s: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """A_i^min = G_i / (d_i * phi_i * s_i)  (Theorem 3, Eq. 712 of the paper)."""
    G = np.asarray(G, dtype=float)
    d = np.asarray(d, dtype=float)
    phi = np.asarray(phi, dtype=float)
    s = np.asarray(s, dtype=float)
    denom = np.maximum(d * phi * s, eps)
    return G / denom
