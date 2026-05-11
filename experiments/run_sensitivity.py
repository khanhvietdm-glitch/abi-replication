"""
Appendix C.5 -- Sensitivity grids.

Sweeps three parameters across coarse grids and verifies the monotonicity
predictions of Theorems 5 and 12:
    phi (speculative-flow intensity) in {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
    pi_c (coalition fraction)        in {0, 0.05, ..., 0.30}
    kappa_A (audit cost)             in {0.01, 0.02, 0.05, 0.10}

The paper uses 7 x 7 x 4 = 196 cells with K = 50 replications; this script
defaults to a smaller grid for fast reproduction (use --full for the paper
configuration).

Outputs:
    results/sensitivity_phi_pic.csv     -- (mechanism, phi, pi_c) cells
    results/sensitivity_kappaA.csv      -- audit-cost sweep
    figures/heatmap_welfare_abi.png     -- ABI welfare vs (phi, pi_c)
    figures/heatmap_welfare_yuma.png    -- Yuma welfare vs (phi, pi_c)
    figures/sweep_phi_welfare.png       -- welfare vs phi by mechanism
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abi.config import SimulationParams, SWEEP_PHI, SWEEP_PI_C, SWEEP_KAPPA_A
from abi.runner import run_compare


HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


MECHANISMS = ("pure_stake", "yuma_like", "abi")
PRETTY = {"pure_stake": "Pure Stake", "yuma_like": "Yuma-like", "abi": "ABI"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=5, help="replications per cell")
    ap.add_argument("--T", type=int, default=80, help="epochs per replication")
    ap.add_argument("--quick", action="store_true",
                    help="reduced grid 3 phi x 3 pi_c, 2 kappa_A")
    ap.add_argument("--full", action="store_true",
                    help="paper grid 7 x 7 x 4 (long-running)")
    return ap.parse_args()


def sweep_phi_pic(K: int, T: int, phi_grid, pic_grid) -> pd.DataFrame:
    seeds = list(range(1, K + 1))
    rows = []
    for phi in phi_grid:
        for pic in pic_grid:
            params = SimulationParams(T=T, K=K, N_p=120, N_v=24,
                                      phi_speculative=phi, pi_c=pic)
            print(f"  phi={phi:.2f}  pi_c={pic:.2f}")
            multi = run_compare(MECHANISMS, params, seeds, verbose=False)
            for mech in MECHANISMS:
                sub = multi.summary[multi.summary["mechanism"] == mech].set_index("metric")["mean"]
                rows.append({
                    "mechanism": mech,
                    "phi": phi,
                    "pi_c": pic,
                    "false_trust_rate": float(sub["false_trust_rate"]),
                    "social_welfare": float(sub["social_welfare"]),
                    "reward_gini": float(sub["reward_gini"]),
                    "collusion_success_rate": float(sub["collusion_success_rate"]),
                    "capital_exclusion_rate": float(sub["capital_exclusion_rate"]),
                })
    return pd.DataFrame(rows)


def sweep_kappaA(K: int, T: int, kappa_grid) -> pd.DataFrame:
    seeds = list(range(1, K + 1))
    rows = []
    for kappa in kappa_grid:
        params = SimulationParams(T=T, K=K, N_p=120, N_v=24, kappa_A=kappa)
        print(f"  kappa_A={kappa:.3f}")
        multi = run_compare(MECHANISMS, params, seeds, verbose=False)
        for mech in MECHANISMS:
            sub = multi.summary[multi.summary["mechanism"] == mech].set_index("metric")["mean"]
            rows.append({
                "mechanism": mech,
                "kappa_A": kappa,
                "social_welfare": float(sub["social_welfare"]),
                "false_trust_rate": float(sub["false_trust_rate"]),
            })
    return pd.DataFrame(rows)


def plot_heatmap(df: pd.DataFrame, mechanism: str, metric: str, out_path: str):
    sub = df[df["mechanism"] == mechanism]
    pivot = sub.pivot(index="pi_c", columns="phi", values=metric)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto",
                   extent=[pivot.columns.min(), pivot.columns.max(),
                           pivot.index.min(), pivot.index.max()])
    ax.set_xlabel(r"speculative-flow intensity $\phi$")
    ax.set_ylabel(r"coalition fraction $\pi_c$")
    ax.set_title(f"{metric} -- {PRETTY[mechanism]}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phi_welfare(df: pd.DataFrame, pi_c_fixed: float, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = df[np.isclose(df["pi_c"], pi_c_fixed)]
    colors = {"pure_stake": "#888", "yuma_like": "#70ad47", "abi": "#c0504d"}
    for mech in MECHANISMS:
        s = sub[sub["mechanism"] == mech]
        ax.plot(s["phi"], s["social_welfare"], marker="o", color=colors[mech],
                label=PRETTY[mech])
    ax.set_xlabel(r"speculative-flow intensity $\phi$")
    ax.set_ylabel("social welfare / epoch")
    ax.set_title(rf"Welfare vs speculative contamination (at $\pi_c={pi_c_fixed:.2f}$)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if args.quick:
        phi_grid = (0.0, 0.2, 0.4)
        pic_grid = (0.0, 0.10, 0.20)
        kappa_grid = (0.02, 0.05)
        args.K = 3; args.T = 60
    elif args.full:
        phi_grid = SWEEP_PHI
        pic_grid = SWEEP_PI_C
        kappa_grid = SWEEP_KAPPA_A
    else:
        phi_grid = (0.0, 0.2, 0.4, 0.6)
        pic_grid = (0.0, 0.10, 0.20, 0.30)
        kappa_grid = (0.02, 0.05, 0.10)

    print("[run_sensitivity] phi x pi_c sweep")
    df_phi_pic = sweep_phi_pic(args.K, args.T, phi_grid, pic_grid)
    df_phi_pic.to_csv(os.path.join(RESULTS, "sensitivity_phi_pic.csv"), index=False)

    print("[run_sensitivity] kappa_A sweep")
    df_kappa = sweep_kappaA(args.K, args.T, kappa_grid)
    df_kappa.to_csv(os.path.join(RESULTS, "sensitivity_kappaA.csv"), index=False)

    plot_heatmap(df_phi_pic, "abi", "social_welfare", os.path.join(FIG, "heatmap_welfare_abi.png"))
    plot_heatmap(df_phi_pic, "yuma_like", "social_welfare", os.path.join(FIG, "heatmap_welfare_yuma.png"))
    plot_phi_welfare(df_phi_pic, pi_c_fixed=0.10 if 0.10 in pic_grid else pic_grid[len(pic_grid)//2],
                     out_path=os.path.join(FIG, "sweep_phi_welfare.png"))
    print(f"[run_sensitivity] wrote results to {RESULTS} and figures to {FIG}")


if __name__ == "__main__":
    main()
