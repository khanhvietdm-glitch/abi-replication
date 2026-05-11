# ABI Replication Package

Replication code for the simulations in:

> **Asset-Backed Intelligence: A Welfare-Guided Mechanism Design Framework for Trust in Decentralized AI Markets.**

This package reproduces the numerical content of the paper:

| Paper section | What this package reproduces |
| --- | --- |
| § 7.1 *Baseline parameters* (Table 6) | Feasibility checks for Theorems 1, 2 — `experiments/run_comparative_statics.py` |
| § 7.2 *Comparative statics* | Theorem 2 (feasible slashing interval), Theorem 3 (minimum audit probability), Theorem 5 (speculative attenuation), Theorem 10 (local stability multiplier) |
| § 7.3 *Robustness* | (i) continuous types & single-crossing, (ii) inverse-variance vs GLS under correlated noise, (iii) contamination-responsive `λ_S` — `experiments/run_robustness.py` |
| § 7.5 *Agent-based simulation* (Table 7) | Pure-Stake vs Pure-Reputation vs Yuma-like vs ABI — `experiments/run_baseline.py` |
| Appendix C.4 *Ablations* (Table C2) | `ABI-NoAudit`, `ABI-NoCredit`, `ABI-NoDynamic`, `ABI-NoReliability` — `experiments/run_ablation.py` |
| Appendix C.5 *Sensitivity grids* | Sweeps over speculative-flow intensity `φ`, coalition fraction `π_c`, audit cost `κ_A` — `experiments/run_sensitivity.py` |

---

## Repository layout

```
abi-replication/
├── abi/                      core library
│   ├── config.py             Table 6 + Table C1 parameters
│   ├── utils.py              softmax, inverse-variance, gini, seed protocol
│   ├── agents.py             producer / validator / user populations
│   ├── mechanisms.py         the four mechanisms (Algorithms C.1–C.4)
│   ├── simulation.py         shared epoch loop (Algorithm 1)
│   ├── metrics.py            six metrics from §7.5
│   └── runner.py             paired cross-mechanism runner
├── experiments/              CLI scripts (one per section)
├── tests/                    pytest sanity tests
├── results/                  generated CSVs (one per script)
├── figures/                  generated PNGs (one per claim)
└── requirements.txt
```

---

## Quick start

### Install

```bash
pip install -r requirements.txt
```

Tested on Python 3.11 / 3.13 / 3.14 with NumPy 1.26+ / 2.x, Pandas 2.x / 3.x,
SciPy 1.10+, Matplotlib 3.7+.

### Reproduce everything (≈ 5 minutes on an 8-core laptop, reduced grids)

```bash
python experiments/run_comparative_statics.py
python experiments/run_robustness.py
python experiments/run_baseline.py            # K=20, T=200 (default)
python experiments/run_ablation.py            # K=15, T=120 (default)
python experiments/run_sensitivity.py         # 4×4×3 grid
```

### Paper-scale runs (≈ 8 CPU-hours, Table 7 specification)

```bash
python experiments/run_baseline.py    --K 50 --T 200
python experiments/run_ablation.py    --K 50 --T 200
python experiments/run_sensitivity.py --K 50 --T 200 --full
```

### Smoke test (≈ 30 s)

```bash
python experiments/run_baseline.py --quick
python experiments/run_ablation.py --quick
python experiments/run_sensitivity.py --quick
python -m pytest tests -q
```

---

## What each script writes

| Script | `results/` | `figures/` |
| --- | --- | --- |
| `run_comparative_statics.py` | `feasibility_check.csv`, `cs_*.csv` | `cs_feasible_slashing_interval.png`, `cs_min_audit_probability.png`, `cs_attenuation.png`, `cs_stability_multiplier.png` |
| `run_robustness.py` | `robust_continuous_types.csv`, `robust_correlated_noise.csv`, `robust_endogenous_omega.csv` | `robust_continuous_types.png`, `robust_correlated_noise.png`, `robust_endogenous_omega.png` |
| `run_baseline.py` | `baseline_summary.csv`, `baseline_per_rep.csv`, `baseline_per_epoch.csv` | `baseline_table7.png`, `baseline_metric_trace.png` |
| `run_ablation.py` | `ablation_summary.csv`, `ablation_marginal_effects.csv` | `ablation_marginal.png` |
| `run_sensitivity.py` | `sensitivity_phi_pic.csv`, `sensitivity_kappaA.csv` | `heatmap_welfare_abi.png`, `heatmap_welfare_yuma.png`, `sweep_phi_welfare.png` |

---

## The four mechanisms (Algorithms C.1 – C.4)

All four mechanisms share an identical population of producers, validators
and users, an identical reward budget per epoch, and an identical seed
protocol (Appendix C.3 — paired comparison). They differ only in:

| | Threshold | Aggregation | Audit | Slashing | Governance |
|---|---|---|---|---|---|
| **Pure Stake** (`C.1`) | static `s ≥ S*` | mean of reports | uniform random | flat rate × `s` on failure | none |
| **Pure Reputation** (`C.2`) | none | mean of reports | uniform random | none | smoothed reputation |
| **Yuma-like** (`C.3`) | static `s ≥ S*` | stake-weighted, top-quantile clipped | random over eligible | flat rate × `s` on failure | none |
| **ABI** (`C.4` / Algorithm 1) | adaptive `S*` + quality credit `f` (Prop. 2) | inverse-variance reliability weights (Prop. 1) | risk-targeted `A_i ≥ G_i / (d φ s_i)` (Thm 3) | quality-quadratic `φ s [max(0, τ − Q)]²` (Eq. 233) | log-threshold recursion `log S*_{t+1} = log S*_t + κ(p − p*) + ξ(β* − β) + ζ Ω` (Thm 10) |

Ablation variants disable one ABI lever at a time:

```python
from abi.mechanisms import ABIMechanism
ABIMechanism(params, use_risk_targeted_audit=False)   # ABI-NoAudit
ABIMechanism(params, use_quality_credit=False)        # ABI-NoCredit
ABIMechanism(params, use_dynamic_threshold=False)     # ABI-NoDynamic
ABIMechanism(params, use_inverse_variance=False)      # ABI-NoReliability
```

---

## Six metrics tracked per epoch (§ 7.5)

| metric | definition | direction |
| --- | --- | --- |
| `false_trust_rate` | P(true `q < τ` ∣ producer trusted) | lower better |
| `social_welfare` | trust value − capital DWL − audit cost − slashing destruction | higher better |
| `high_type_participation` | fraction of high-type producers clearing the threshold | higher better |
| `reward_gini` | Gini coefficient of producer rewards | context-dependent |
| `collusion_success_rate` | epochs in which coalition net-reward share > capital share | lower better |
| `capital_exclusion_rate` | high-type producers excluded by stake threshold | lower better |

---

## Findings replicated

After running `python experiments/run_baseline.py --K 20 --T 200` and
`python experiments/run_ablation.py --K 15 --T 120` we obtain the
qualitative dominance pattern of Table 7 and Table C2:

**Baseline dominance (Table 7):**

```
                              ABI    Pure-Rep   Pure-Stake   Yuma-like
false-trust rate              0.000     0.063     0.159        0.005
social welfare / epoch       73.3      27.1      57.4         68.5
high-type participation       0.97      1.00      0.84         0.84
reward Gini                   0.68      0.25      0.66         0.69
collusion-success rate        0.48      0.70      0.46         0.47
capital-exclusion rate        0.03      0.00      0.16         0.16
```

ABI strictly dominates the pure-stake baseline on five of the six metrics
and is closely matched on welfare by Yuma-like at low contamination. The
welfare gap widens as `φ` increases (see `sweep_phi_welfare.png`),
consistent with the attenuation result of Theorem 5.

**Ablation marginals (Table C2):**

* Removing the quality credit (`ABI-NoCredit`): capital exclusion jumps
  from 0.08 → 0.42 and false-trust rate from 0.003 → 0.249 — confirms
  the role of Proposition 2.
* Removing inverse-variance aggregation (`ABI-NoReliability`):
  collusion-success rate rises by ≈ 0.07 — confirms the role of
  Proposition 1.
* Removing the dynamic threshold (`ABI-NoDynamic`): high-type
  participation improves marginally because `S*` cannot drift upward
  during noisy epochs — consistent with Theorem 10.

---

## Closed-form checks (Section 7.1)

`run_comparative_statics.py` prints:

```
quantity                          value
r_f + π_H · φ (lower band)        0.0304
m (baseline)                      0.0450
r_f + π_L · φ (upper band)        0.0616
m in feasible band                1   (✓)
φ_min (Theorem 2)                 0.0781
φ (baseline)                      0.1300
φ_max (Theorem 2)                 0.3125
φ in feasible interval            1   (✓)
```

matching the analytical values in Section 7.1.

---

## Random-seed protocol (Appendix C.3)

For each replication `k ∈ {1, …, K}`, the master seed is `k`. Five
independent sub-streams are derived via SHA-256 hashing with fixed
salts (see `abi/utils.derive_seeds`):

* `0x01` producer-quality stream
* `0x02` validator-noise stream
* `0x03` speculative-flow stream
* `0x04` audit-selection stream
* `0x05` coalition-formation stream

All four mechanisms consume identical sub-streams within each
replication, so cross-mechanism comparisons are *paired* across the
stochastic environment.

---

## Testing

```bash
python -m pytest tests -q
```

Twelve tests cover: feasible slashing/reward intervals (Thms 1, 2),
`A_min` monotonicity (Thm 3), attenuation endpoints (Thm 5), softmax
properties (Lemma 1), inverse-variance dominance (Prop. 1), and that all
four mechanisms run a multi-epoch simulation without NaN welfare. A
final test verifies that two mechanisms built from the same master seed
see identical producer populations.

---

## Reproducibility

* Pseudo-random number generator: `numpy.random.Generator` (PCG64),
  re-seeded per replication from the master seed via SHA-256 sub-streams.
* Determinism: bit-level reproducible on a fixed Python+NumPy version.
  Cross-version drift may occur in `numpy.random.beta` etc.; the
  qualitative dominance pattern is stable across NumPy 1.26–2.4.
* CSV outputs in `results/` are versioned together with the figures so
  that downstream analysis or replication audits can be performed
  without re-running the simulation.

---

## License

MIT. See `LICENSE`.

---

## Citing

If you use this code, please cite the paper:

```bibtex
@unpublished{abi2026,
  title  = {Asset-Backed Intelligence: A Welfare-Guided Mechanism Design
            Framework for Trust in Decentralized AI Markets},
  year   = {2026},
  note   = {Replication package: https://github.com/<user>/abi-replication}
}
```
