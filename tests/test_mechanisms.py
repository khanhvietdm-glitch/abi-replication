"""
Minimal sanity tests for the four mechanisms and the closed-form formulas.
Run with:  python -m pytest tests
"""
from __future__ import annotations

import numpy as np
import pytest

from abi.config import SimulationParams, ABI_BASELINE
from abi.mechanisms import build_mechanism
from abi.simulation import Simulator
from abi.utils import (
    feasible_slashing_interval,
    feasible_marginal_reward_band,
    min_audit_probability,
    attenuated_elasticity,
    softmax_rewards,
    inverse_variance_weights,
)


def test_baseline_phi_in_feasible_interval():
    c = ABI_BASELINE
    phi_lo, phi_hi = feasible_slashing_interval(c.m, c.r_f, c.pi_H, c.pi_L)
    assert phi_lo <= c.phi <= phi_hi


def test_baseline_m_in_feasible_band():
    c = ABI_BASELINE
    lo, hi = feasible_marginal_reward_band(c.phi, c.r_f, c.pi_H, c.pi_L)
    assert lo <= c.m <= hi


def test_min_audit_monotone():
    G = np.array([0.05])
    A1 = min_audit_probability(G, np.array([0.5]), np.array([0.10]), np.array([1.0]))
    A2 = min_audit_probability(G, np.array([0.9]), np.array([0.10]), np.array([1.0]))
    assert A2 < A1  # higher detection -> lower audit probability


def test_attenuation_zero_at_omega_one():
    assert attenuated_elasticity(0.5, 1.0) == 0.0


def test_attenuation_full_at_omega_zero():
    assert attenuated_elasticity(0.5, 0.0) == pytest.approx(0.5)


def test_softmax_sum_equals_budget():
    z = np.array([0.1, 0.4, 0.2, 0.7])
    r = softmax_rewards(z, eta=3.0, total_budget=100.0)
    assert r.sum() == pytest.approx(100.0)


def test_inverse_variance_weights_sum_to_one():
    sigma2 = np.array([0.04, 0.09, 0.01])
    w = inverse_variance_weights(sigma2)
    assert w.sum() == pytest.approx(1.0)
    # validator with smallest variance gets largest weight
    assert np.argmax(w) == np.argmin(sigma2)


@pytest.mark.parametrize("mech_name", ["pure_stake", "pure_reputation", "yuma_like", "abi"])
def test_mechanism_runs_one_epoch(mech_name):
    p = SimulationParams(N_p=30, N_v=8, N_u=200, T=3, burn_in=0, K=1)
    mech = build_mechanism(mech_name, p)
    sim = Simulator(p, mech, master_seed=1, total_reward_budget=50.0)
    result = sim.run()
    assert result.per_epoch.shape[0] == p.T
    # Welfare should not be NaN
    assert not result.per_epoch["social_welfare"].isna().any()


def test_paired_seed_yields_same_quality_draws():
    """Two mechanisms with the same seed see the same producer-quality stream."""
    p = SimulationParams(N_p=50, N_v=10, N_u=200, T=5, burn_in=0)
    sims = []
    for name in ["pure_stake", "abi"]:
        m = build_mechanism(name, p)
        s = Simulator(p, m, master_seed=42)
        sims.append(s)
    # Both Simulators built ProducerPopulations from the same master seed.
    np.testing.assert_array_equal(sims[0].prod.type_high, sims[1].prod.type_high)
    np.testing.assert_allclose(sims[0].prod.capital_endow, sims[1].prod.capital_endow)
