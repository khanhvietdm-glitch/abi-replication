"""
Section 7.5 -- Agent-based simulation, baseline scenario (reproduces Table 7).

Runs the four mechanisms (Pure-Stake, Pure-Reputation, Yuma-like, ABI) under
identical population draws via the paired-seed protocol of Appendix C.3.

Outputs:
    results/baseline_summary.csv      -- mean +/- 95% CI per (mechanism, metric)
    results/baseline_per_rep.csv      -- one row per (mechanism, seed)
    results/baseline_per_epoch.csv    -- full per-epoch trace
    figures/baseline_table7.png       -- bar chart reproducing Table 7
    figures/baseline_metric_trace.png -- per-epoch trace for each metric
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abi.config import SimulationParams
from abi.runner import run_compare


HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


MECHANISMS = ("pure_stake", "pure_reputation", "yuma_like", "abi")

PRETTY = {
    "pure_stake": "Pure Stake",
    "pure_reputation": "Pure Reputation",
    "yuma_like": "Yuma-like",
    "abi": "ABI (this paper)",
}

METRIC_PRETTY = {
    "false_trust_rate": "False-trust rate",
    "social_welfare": "Social welfare / epoch",
    "high_type_participation": "High-type participation",
    "reward_gini": "Reward Gini",
    "collusion_success_rate": "Collusion success rate",
    "capital_exclusion_rate": "Capital-exclusion rate",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=20, help="number of replications (paper uses 50)")
    ap.add_argument("--T", type=int, default=200, help="epochs per replication")
    ap.add_argument("--N_p", type=int, default=200)
    ap.add_argument("--N_v", type=int, default=40)
    ap.add_argument("--N_u", type=int, default=5000)
    ap.add_argument("--phi", type=float, default=0.20, help="speculative-flow intensity")
    ap.add_argument("--pi_c", type=float, default=0.10, help="coalition fraction")
    ap.add_argument("--quick", action="store_true", help="K=4, T=60 for smoke test")
    return ap.parse_args()


def plot_table7(summary: pd.DataFrame, out_path: str):
    pivot = summary.pivot(index="metric", columns="mechanism", values="mean")
    ci = summary.pivot(index="metric", columns="mechanism", values="ci95")
    metrics = ["false_trust_rate", "social_welfare", "high_type_participation",
               "reward_gini", "collusion_success_rate", "capital_exclusion_rate"]
    pivot = pivot.reindex(metrics)
    ci = ci.reindex(metrics)
    mechanisms = ["pure_stake", "pure_reputation", "yuma_like", "abi"]
    pivot = pivot[mechanisms]
    ci = ci[mechanisms]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, metric in zip(axes.flat, metrics):
        vals = pivot.loc[metric].to_numpy()
        cis = ci.loc[metric].to_numpy()
        x = np.arange(len(vals))
        colors = ["#888", "#5b9bd5", "#70ad47", "#c0504d"]
        ax.bar(x, vals, yerr=cis, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY[m] for m in mechanisms], rotation=18, fontsize=8)
        ax.set_title(METRIC_PRETTY[metric])
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Section 7.5 / Table 7 -- Agent-based simulation, baseline scenario", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_trace(per_epoch: pd.DataFrame, out_path: str):
    metrics = ["false_trust_rate", "social_welfare", "reward_gini", "capital_exclusion_rate"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    colors = {"pure_stake": "#888", "pure_reputation": "#5b9bd5",
              "yuma_like": "#70ad47", "abi": "#c0504d"}
    for ax, metric in zip(axes.flat, metrics):
        for mech, g in per_epoch.groupby("mechanism"):
            traj = g.groupby("epoch")[metric].mean()
            ax.plot(traj.index, traj.values, color=colors[mech], label=PRETTY[mech], linewidth=1.0)
        ax.set_title(METRIC_PRETTY[metric])
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Per-epoch trace (mean across replications)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if args.quick:
        args.K = 4
        args.T = 60
        args.N_p = 80
        args.N_v = 20

    params = SimulationParams(
        N_p=args.N_p, N_v=args.N_v, N_u=args.N_u,
        T=args.T, K=args.K,
        phi_speculative=args.phi, pi_c=args.pi_c,
    )
    seeds = list(range(1, args.K + 1))

    print(f"[run_baseline] running {len(MECHANISMS)} mechanisms x {args.K} seeds x {args.T} epochs")
    multi = run_compare(MECHANISMS, params, seeds, total_reward_budget=100.0, verbose=True)

    summary = multi.summary
    summary.to_csv(os.path.join(RESULTS, "baseline_summary.csv"), index=False)
    multi.per_replication.to_csv(os.path.join(RESULTS, "baseline_per_rep.csv"), index=False)
    multi.per_epoch_long.to_csv(os.path.join(RESULTS, "baseline_per_epoch.csv"), index=False)

    pivot = summary.pivot(index="metric", columns="mechanism", values="mean").round(4)
    print("\n[run_baseline] mean values (baseline scenario):")
    print(pivot.to_string())

    plot_table7(summary, os.path.join(FIG, "baseline_table7.png"))
    plot_metric_trace(multi.per_epoch_long, os.path.join(FIG, "baseline_metric_trace.png"))
    print(f"\n[run_baseline] wrote results to {RESULTS} and figures to {FIG}")


if __name__ == "__main__":
    main()
