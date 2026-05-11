"""
Section 7.3 -- Robustness checks (relaxing core assumptions).

7.3.1 Continuous types -- replace binary {H, L} with a continuous type
      drawn from Beta(alpha, beta) and verify monotone stake schedule.

7.3.2 Correlated validator noise -- replace inverse-variance weighting
      (Proposition 1) with the generalized inverse-covariance estimator
      (Eq. 167), and compare aggregator MSE.

7.3.3 Endogenous Omega -- let speculative contamination respond to
      staking participation; verify contamination-responsive lambda_S rule.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


# ---------------------------------------------------------------------------
# 7.3.1 -- Continuous types and monotone stake schedule
# ---------------------------------------------------------------------------
def continuous_type_stake_schedule(n: int = 1000, m: float = 0.045, r_f: float = 0.020,
                                   phi: float = 0.13, seed: int = 17) -> tuple:
    """
    Producer utility U(s,theta) = R(s,theta) - r_f s - pi(theta) phi s.
    Under R(s,theta) = a_theta + m s and pi(theta) = pi_max - alpha theta, the
    optimal s* solves dU/ds = 0 -> requires single-crossing. Numerical solve
    on a grid for illustration.
    """
    rng = np.random.default_rng(seed)
    theta = rng.beta(2.0, 2.0, size=n)         # continuous quality
    pi_max, pi_min = 0.40, 0.05
    pi_theta = pi_max - (pi_max - pi_min) * theta   # higher theta -> lower failure
    # Net marginal return per unit stake
    marginal = m - r_f - pi_theta * phi
    # Best response: stake monotone in marginal (clip at zero)
    s_star = np.maximum(0.0, marginal) * 10.0
    df = pd.DataFrame({"theta": theta, "pi_theta": pi_theta, "marginal_return": marginal, "s_star": s_star})
    # Check monotonicity by Spearman correlation
    rho = float(pd.Series(s_star).corr(pd.Series(theta), method="spearman"))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    order = np.argsort(theta)
    ax.scatter(theta, s_star, alpha=0.3, s=8, color="#4c72b0")
    ax.plot(theta[order], s_star[order], color="#c0504d", linewidth=1.0,
            label=f"Spearman rho = {rho:.3f}")
    ax.set_xlabel(r"continuous type $\theta$")
    ax.set_ylabel(r"optimal stake $s^*(\theta)$")
    ax.set_title("7.3.1 Continuous types -- monotone stake schedule")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "robust_continuous_types.png"), dpi=150)
    plt.close(fig)
    return df, rho


# ---------------------------------------------------------------------------
# 7.3.2 -- Correlated validator noise: GLS vs inverse-variance
# ---------------------------------------------------------------------------
def correlated_noise_comparison(n_validators: int = 12, n_trials: int = 3000,
                                rho_grid=(0.0, 0.2, 0.4, 0.6, 0.8), seed: int = 42) -> tuple:
    """
    Generate validator errors with equicorrelated covariance Sigma_ij = sigma^2(rho + (1-rho)*1_{i=j}).
    Compare:
        (a) inverse-variance weights w_j ∝ 1/sigma_j^2 (Proposition 1)
        (b) generalized weights w* = Sigma^{-1} 1 / 1^T Sigma^{-1} 1 (Eq. 167)
    on aggregator MSE relative to the truth q = 0.5.
    """
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.05, 0.30, size=n_validators)
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for rho in rho_grid:
        # Equicorrelated correlation matrix
        R = np.full((n_validators, n_validators), rho)
        np.fill_diagonal(R, 1.0)
        Sigma = R * np.outer(sigma, sigma)
        # Inverse-variance weights
        w_iv = (1.0 / sigma ** 2)
        w_iv /= w_iv.sum()
        # GLS weights
        try:
            inv = np.linalg.inv(Sigma)
            w_gls = inv @ np.ones(n_validators)
            w_gls /= w_gls.sum()
        except np.linalg.LinAlgError:
            w_gls = w_iv.copy()
        # Simulate trials
        # Sampling from N(0, Sigma) via Cholesky
        L = np.linalg.cholesky(Sigma + 1e-9 * np.eye(n_validators))
        errs = rng.standard_normal((n_trials, n_validators)) @ L.T
        q_true = 0.5
        y = q_true + errs
        mse_iv = float(np.mean((y @ w_iv - q_true) ** 2))
        mse_gls = float(np.mean((y @ w_gls - q_true) ** 2))
        rows.append({"rho_corr": rho, "mse_inverse_variance": mse_iv, "mse_gls": mse_gls})
    df = pd.DataFrame(rows)
    ax.plot(df["rho_corr"], df["mse_inverse_variance"], marker="o", color="#5b9bd5", label="Inverse-variance (Prop. 1)")
    ax.plot(df["rho_corr"], df["mse_gls"], marker="s", color="#c0504d", label="GLS / inverse-covariance (Eq. 167)")
    ax.set_xlabel(r"validator-error correlation $\rho$")
    ax.set_ylabel("aggregator MSE")
    ax.set_title("7.3.2 Correlated validator noise -- aggregator MSE")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "robust_correlated_noise.png"), dpi=150)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# 7.3.3 -- Endogenous Omega and contamination-responsive lambda_S
# ---------------------------------------------------------------------------
def endogenous_omega(p_grid=np.linspace(0.0, 1.0, 51)) -> tuple:
    """
    Reduced-form: Omega(p) = sigmoid(5*(p - 0.6)).
    Compare two lambda_S rules:
      fixed lambda_S0 = 0.20
      adaptive lambda_S(t) = lambda_S^max * (1 - Omega_t)
    Plot lambda_S(t) and resulting stake-weight in trust score.
    """
    Omega = 1.0 / (1.0 + np.exp(-5.0 * (p_grid - 0.6)))
    lam_fixed = np.full_like(p_grid, 0.20)
    lam_adapt = 0.35 * (1.0 - Omega)
    lam_floor = 0.05 + 0.30 * np.clip(1.0 - Omega, 0.0, 1.0)
    df = pd.DataFrame({"p": p_grid, "Omega": Omega,
                       "lambda_S_fixed": lam_fixed,
                       "lambda_S_adaptive": lam_adapt,
                       "lambda_S_adaptive_floor": lam_floor})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p_grid, Omega, color="black", linestyle=":", label=r"$\Omega(p)$ (contamination)")
    ax.plot(p_grid, lam_fixed, color="#5b9bd5", label=r"$\lambda_S$ fixed (0.20)")
    ax.plot(p_grid, lam_adapt, color="#c0504d", label=r"$\lambda_S^{\max}(1-\Omega)$")
    ax.plot(p_grid, lam_floor, color="#70ad47", label=r"floored adaptive rule")
    ax.set_xlabel(r"staking participation $p$")
    ax.set_ylabel("value")
    ax.set_title("7.3.3 Contamination-responsive stake weight $\\lambda_S$")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "robust_endogenous_omega.png"), dpi=150)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
def main():
    print("[run_robustness] 7.3.1 continuous types...")
    df_cont, rho = continuous_type_stake_schedule()
    df_cont.to_csv(os.path.join(RESULTS, "robust_continuous_types.csv"), index=False)
    print(f"  spearman(theta, s*) = {rho:.3f}  (should be ~ 1 under single-crossing)")

    print("[run_robustness] 7.3.2 correlated noise...")
    df_corr = correlated_noise_comparison()
    df_corr.to_csv(os.path.join(RESULTS, "robust_correlated_noise.csv"), index=False)
    print(df_corr.round(5).to_string(index=False))

    print("[run_robustness] 7.3.3 endogenous Omega...")
    df_om = endogenous_omega()
    df_om.to_csv(os.path.join(RESULTS, "robust_endogenous_omega.csv"), index=False)
    print(f"[run_robustness] wrote results to {RESULTS} and figures to {FIG}")


if __name__ == "__main__":
    main()
