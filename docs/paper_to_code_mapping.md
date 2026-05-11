# Paper-to-code mapping

Every theorem, proposition, lemma, and parameter table that has a
numerical or simulation footprint is wired to a specific module.

## Theorems and propositions

| Paper claim | Code reference |
| --- | --- |
| **Theorem 1** stake-based separating equilibrium: `r_f + π_H φ ≤ m ≤ r_f + π_L φ` | `abi/utils.py :: feasible_marginal_reward_band` — used in `run_comparative_statics.baseline_feasibility` |
| **Theorem 2** feasible slashing interval: `φ ∈ [(m−r_f)/π_L, (m−r_f)/π_H]` | `abi/utils.py :: feasible_slashing_interval` — used in `run_comparative_statics.comparative_slashing_interval` |
| **Theorem 3** minimum audit probability: `A_i ≥ G_i / (d_i φ_i s_i)` | `abi/utils.py :: min_audit_probability`; risk-targeted audit hook in `ABIMechanism.select_audit` |
| **Theorem 4** information ceiling on binary stake: `I(B;Q) ≤ h_2(p)` | `abi/utils.py :: binary_entropy` (and discussed in `experiments/run_robustness.py`) |
| **Theorem 5** speculative attenuation: `β_obs = (1 − Ω) β_commit` | `abi/utils.py :: attenuated_elasticity`; `abi/simulation.py :: _contamination_ratio` |
| **Theorem 6** saturation of marginal stake utility, `g'(x) → 0` | `abi/utils.py :: stake_transform` (`g(x) = log(1+x)`) |
| **Theorem 10** local stability of `log S*` recursion | `abi/utils.py :: update_log_threshold`; called from `ABIMechanism.governance_update` |
| **Theorem 12** coalition resistance: `G_C^eff ≤ Σ A_i d_i φ_i s_i + Σ A_j d_j φ_j b_j` | `abi/simulation.py :: _coalition_gain_and_slash`; metric in `abi/metrics.py :: _collusion_success` |
| **Lemma 1** softmax reward gradient | `abi/utils.py :: softmax_rewards`; used by `ABIMechanism.allocate_rewards` |
| **Proposition 1** inverse-variance validator aggregation | `abi/utils.py :: inverse_variance_weights, inverse_variance_estimate`; used by `ABIMechanism.aggregate_reports` |
| **Proposition 2** capital fairness via clawback-eligible quality credit | `abi/utils.py :: quality_credit`; called from `ABIMechanism.update_quality_credit` |
| **Proposition 3** information-theoretic necessity for low harmful trust (Fano) | `abi/utils.py :: binary_entropy` (used in commentary plots) |

## Algorithms

| Paper algorithm | Code reference |
| --- | --- |
| **Algorithm 1** ABI epoch loop (§ 8.1) | Inner loop of `abi/simulation.Simulator.run`, hooks in `abi/mechanisms.ABIMechanism` |
| **Algorithm C.1** Pure-Stake baseline | `abi/mechanisms.PureStakeMechanism` |
| **Algorithm C.2** Pure-Reputation baseline | `abi/mechanisms.PureReputationMechanism` |
| **Algorithm C.3** Yuma-like baseline (stylized) | `abi/mechanisms.YumaLikeMechanism` |
| **Algorithm C.4** ABI mechanism | `abi/mechanisms.ABIMechanism` |

## Parameter tables

| Table | Code reference |
| --- | --- |
| **Table 6** ABI illustrative calibration | `abi/config.ABICalibration` (used by `run_comparative_statics`) |
| **Table C1** agent-based simulation parameters | `abi/config.SimulationParams` |
| **Table C2** ablation variants | `abi/mechanisms.build_mechanism` keys: `abi_no_audit`, `abi_no_credit`, `abi_no_dynamic`, `abi_no_reliability` |

## Random-seed protocol (Appendix C.3)

`abi/utils.derive_seeds(master_seed)` returns SHA-256-derived sub-stream
seeds with the same salt mapping (`0x01`–`0x05`) as the paper.

## Sensitivity grids (Appendix C.5)

`abi/config.SWEEP_PHI`, `abi/config.SWEEP_PI_C`, `abi/config.SWEEP_KAPPA_A`
contain the seven-by-seven-by-four grid; activate with
`run_sensitivity.py --full`.
