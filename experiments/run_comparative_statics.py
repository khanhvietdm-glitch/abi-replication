"""
Section 7.2 -- Illustrative numerical calibration and comparative statics.

Reproduces:
    * Theorem 1 feasibility check          (Eq. r_f + pi_H phi <= m <= r_f + pi_L phi)
    * Theorem 2 feasible slashing interval (Eq. phi in [(m - r_f)/pi_L, (m - r_f)/pi_H])
    * Theorem 3 minimum audit probability  (Eq. A_min = G / (d phi s))
    * Theorem 5 attenuation                 (Eq. beta_obs = (1 - Omega) beta_commit)
    * Theorem 10 local stability multiplier (|1 + g'(x*)| < 1)

Outputs:
    results/comparative_statics_summary.csv  -- numerical values matching Section 7.1
    figures/cs_*.png                         -- one figure per comparative-statics block
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# allow running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abi.config import ABI_BASELINE
from abi.utils import (
    feasible_marginal_reward_band,
    feasible_slashing_interval,
    min_audit_probability,
    attenuated_elasticity,
)


HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------------------
# 7.1 -- baseline feasibility check
# ---------------------------------------------------------------------------
def baseline_feasibility() -> pd.DataFrame:
    c = ABI_BASELINE
    lo, hi = feasible_marginal_reward_band(c.phi, c.r_f, c.pi_H, c.pi_L)
    phi_lo, phi_hi = feasible_slashing_interval(c.m, c.r_f, c.pi_H, c.pi_L)
    rows = [
        {"quantity": "r_f + pi_H * phi (lower band)", "value": lo},
        {"quantity": "m (baseline)", "value": c.m},
        {"quantity": "r_f + pi_L * phi (upper band)", "value": hi},
        {"quantity": "m in feasible band", "value": float(lo <= c.m <= hi)},
        {"quantity": "phi_min (Theorem 2)", "value": phi_lo},
        {"quantity": "phi (baseline)", "value": c.phi},
        {"quantity": "phi_max (Theorem 2)", "value": phi_hi},
        {"quantity": "phi in feasible interval", "value": float(phi_lo <= c.phi <= phi_hi)},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7.2.1 -- type-wedge and feasible slashing interval
# ---------------------------------------------------------------------------
def comparative_slashing_interval() -> tuple:
    """phi-min and phi-max as pi_L varies, holding pi_H, m, r_f fixed."""
    c = ABI_BASELINE
    pi_L_grid = np.linspace(0.10, 0.60, 51)
    phi_min = (c.m - c.r_f) / pi_L_grid
    phi_max = np.full_like(pi_L_grid, (c.m - c.r_f) / c.pi_H)
    df = pd.DataFrame({"pi_L": pi_L_grid, "phi_min": phi_min, "phi_max": phi_max})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pi_L_grid, phi_min, label=r"$\phi_{\min}=(m-r_f)/\pi_L$")
    ax.plot(pi_L_grid, phi_max, label=r"$\phi_{\max}=(m-r_f)/\pi_H$", linestyle="--")
    ax.fill_between(pi_L_grid, phi_min, phi_max, alpha=0.2, label="feasible region")
    ax.axhline(c.phi, color="red", linewidth=0.8, label=f"baseline phi={c.phi}")
    ax.set_xlabel(r"low-type audit-failure probability $\pi_L$")
    ax.set_ylabel(r"slashing intensity $\phi$")
    ax.set_title("Theorem 2 -- Feasible slashing interval as $\\pi_L$ varies")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cs_feasible_slashing_interval.png"), dpi=150)
    plt.close(fig)
    return df, "cs_feasible_slashing_interval.png"


# ---------------------------------------------------------------------------
# 7.2.2 -- minimum audit probability A_min = G/(d phi s)
# ---------------------------------------------------------------------------
def comparative_min_audit() -> tuple:
    c = ABI_BASELINE
    G = 0.05
    d_grid = np.linspace(0.5, 1.0, 51)
    phi_grid = np.linspace(0.05, 0.40, 51)
    s_grid = np.linspace(0.1, 5.0, 51)

    # 1-D slices holding two of {d, phi, s} fixed.
    rows = []
    A_d = G / (d_grid * c.phi * 1.0)
    A_phi = G / (c.d * phi_grid * 1.0)
    A_s = G / (c.d * c.phi * s_grid)
    for x, y, axis in [(d_grid, A_d, "d"), (phi_grid, A_phi, "phi"), (s_grid, A_s, "s")]:
        for xi, yi in zip(x, y):
            rows.append({"axis": axis, "x": float(xi), "A_min": float(yi)})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(d_grid, A_d); axes[0].set_xlabel("d"); axes[0].set_title(r"$A_{\min}$ vs $d$")
    axes[1].plot(phi_grid, A_phi); axes[1].set_xlabel(r"$\phi$"); axes[1].set_title(r"$A_{\min}$ vs $\phi$")
    axes[2].plot(s_grid, A_s); axes[2].set_xlabel("s"); axes[2].set_title(r"$A_{\min}$ vs $s$")
    for ax in axes:
        ax.set_ylabel(r"$A_{\min}=G/(d\phi s)$")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Theorem 3 -- Comparative statics of the minimum audit probability")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cs_min_audit_probability.png"), dpi=150)
    plt.close(fig)
    return df, "cs_min_audit_probability.png"


# ---------------------------------------------------------------------------
# 7.2.3 -- speculative attenuation beta_obs = (1 - Omega) beta_commit
# ---------------------------------------------------------------------------
def comparative_attenuation() -> tuple:
    beta_commit_grid = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    Omega = np.linspace(0.0, 0.9, 50)
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for b in beta_commit_grid:
        beta_obs = attenuated_elasticity(b, Omega)
        ax.plot(Omega, beta_obs, label=rf"$\beta_{{commit}}={b:.1f}$")
        for om, bo in zip(Omega, beta_obs):
            rows.append({"beta_commit": float(b), "Omega": float(om), "beta_observed": float(bo)})
    ax.set_xlabel(r"speculative contamination $\Omega$")
    ax.set_ylabel(r"observed stake-quality elasticity $\beta_{\rm obs}$")
    ax.set_title("Theorem 5 -- Attenuation under speculative contamination")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cs_attenuation.png"), dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows), "cs_attenuation.png"


# ---------------------------------------------------------------------------
# 7.2.4 -- governance gains and local-stability multiplier
# ---------------------------------------------------------------------------
def comparative_stability() -> tuple:
    """
    Local-stability multiplier Lambda = 1 + g'(x*).
    Toy parameterization with p'(x*) ~= -0.5, beta'(x*) ~= -0.2, Omega'(x*) ~= +0.1.
    """
    p_prime = -0.5
    b_prime = -0.2
    o_prime = +0.1
    kappa = np.linspace(0.0, 1.0, 60)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rows = []
    for xi, zeta in [(0.10, 0.10), (0.20, 0.30), (0.40, 0.50), (0.80, 0.80)]:
        g_prime = kappa * p_prime - xi * b_prime + zeta * o_prime
        Lambda = 1.0 + g_prime
        ax.plot(kappa, Lambda, label=fr"$\xi={xi:.2f},\zeta={zeta:.2f}$")
        for k, L in zip(kappa, Lambda):
            rows.append({"kappa": float(k), "xi": xi, "zeta": zeta, "Lambda": float(L)})
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    ax.axhline(-1.0, color="black", linewidth=0.7, linestyle=":")
    ax.axhspan(-1.0, 1.0, alpha=0.1, color="green", label="stable region |Lambda|<1")
    ax.set_xlabel(r"governance gain $\kappa$")
    ax.set_ylabel(r"$\Lambda = 1 + g'(x^*)$")
    ax.set_title("Theorem 10 -- Local-stability multiplier of the threshold recursion")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cs_stability_multiplier.png"), dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows), "cs_stability_multiplier.png"


# ---------------------------------------------------------------------------
def main():
    print("[run_comparative_statics] starting...")

    feas = baseline_feasibility()
    feas.to_csv(os.path.join(RESULTS, "feasibility_check.csv"), index=False)
    print("[run_comparative_statics] feasibility:")
    print(feas.to_string(index=False))

    df_phi, fig_phi = comparative_slashing_interval()
    df_phi.to_csv(os.path.join(RESULTS, "cs_feasible_slashing_interval.csv"), index=False)

    df_a, fig_a = comparative_min_audit()
    df_a.to_csv(os.path.join(RESULTS, "cs_min_audit_probability.csv"), index=False)

    df_b, fig_b = comparative_attenuation()
    df_b.to_csv(os.path.join(RESULTS, "cs_attenuation.csv"), index=False)

    df_s, fig_s = comparative_stability()
    df_s.to_csv(os.path.join(RESULTS, "cs_stability_multiplier.csv"), index=False)

    print(f"[run_comparative_statics] figures saved: {fig_phi}, {fig_a}, {fig_b}, {fig_s}")
    print("[run_comparative_statics] done.")


if __name__ == "__main__":
    main()
