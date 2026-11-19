# ASCA v0: Agentic Structural Causal Attribution (first benchmark)

Every number here comes from `run_benchmark.py` and `run_simulation.py`. The raw output is in `results/`.

## Data used
* **Primary:** Hugging Face `lucianfialho/synthetic-attribution-benchmark` v2 (CC BY 4.0), downloaded directly from the hub into `data/`.
  The full set has 500,000 journeys, 1,349,161 touchpoints and 29,938 conversions (5.99%).
  * **Subsample:** 100,000 journeys stratified on `converted`, seed 7. That gives 5,988 conversions (5.988% vs 5.988% in the full set) and 269,817 touchpoints.
  * **Data-generating process (DGP), checked in code:** `p = σ(−3.2 + Σ_c w_c · s(D_c))`, with `s(x) = x/(1+x)` and `D_c = Σ exp(−ln2/168h · (720 − t))`.
    This reproduces `conv_prob` to 6e-7. For every converting journey, Σ_c gt_shapley_c = conv_prob − σ(−3.2), with error 5e-7.
    Channel weights: paid_search 1.1, social 0.7, display 0.35, email 0.9, organic_search 0.8, direct 1.0, referral 0.5.
  * **Truth used for scoring:** `aggregate_true_contribution` from `ground_truth.json`, built from per-journey normalized Shapley values with each conversion weighted equally.
    I also score against the same quantity recomputed on the subsample.
  * **Caveat:** `dda_comparison_results.json` in the repo has different "gt_shapley" numbers. They look like the v1 aggregation convention, so I did not use them.
* **Agentic simulation:** a self-built simulation with an explicit agent node (`run_simulation.py`). It has 200,000 journeys and 17,907 conversions (8.95%).
  * 50.1% of conversions are agent-mediated (A = 1). 37.5% of conversions show an observed `agent_checkout`.

## Models
Every baseline produces credit per converting journey. Each journey's credit is normalized to 1, then averaged across journeys, which matches the ground-truth convention.
* **Last-click, first-click, linear.**
* **Time-decay:** half-life 168h, measured back from t = 720h.
* **Position-based:** 40/20/40.
* **Markov:** first-order chain with removal effect. The aggregate share is the normalized removal effect. Per journey, the removal effect is spread across the channels present.
* **Shapley-Zhao (counts):** uses v(S) = Σ_{T⊆S} conv(T). This reduces to an equal split across the distinct channels in a journey.
* **Shapley set-CR:** for each converting journey, exact Shapley with v(S) = the empirical conversion rate of journeys whose exact channel set is S. Here v(∅) = 0, unseen sets count as 0, and negative values are clipped.
* **ASCA v0** (`asca/model.py`):
  * Exposure: E_jc = Σ exp(−ln2/h · (T_end − t)). The half-life h is a free parameter.
  * Saturation: s = E/(κ+E). κ is a free parameter.
  * Latent agent node: logit P(A=1|X,Z) = a0 + g·s + zA·Z.
  * Outcome: logit P(Y=1|X,A,Z) = b0 + b·s + d·A + zY·Z.
  * Agent proxy (optional): M | Y=1, A ~ Bern(q·A). This is the observed AI-executed checkout. It is a *descendant* of A and Y, never an exposure.
  * A is marginalized out of the likelihood.
  * Inference: MAP with priors N(0, 2²) on effects, N(0, 5²) on intercepts, log h ~ N(log 100, 2²) and log κ ~ N(0, 2²). Uncertainty comes from a Laplace approximation (finite-difference Hessian), using 200 posterior draws.
  * Credit: v_j(S) = E[Y | do(X_S = x_jS, X_−S = 0), Z_j]. φ_j is the exact Shapley value over all 2^K coalitions. Rows are normalized and averaged.
  * Incremental conversions: Σ_j [v_j(all) − v_j(∅)], allocated by the unnormalized Shapley values.
  * Path-specific credit: the "through A" fraction is 1 − φ^dir/φ^tot, where φ^dir holds the A-equation at X = 0.

## Results: benchmark (100k subsample, scored against the 500k truth)
| model | MAE | MaxAE | Spearman | Kendall | per-journey MAE |
|---|---|---|---|---|---|
| ASCA v0 (latent A) | 0.0057 | 0.0098 | 1.000 | 1.000 | 0.0069 |
| ASCA v0, no A | 0.0057 | 0.0098 | 1.000 | 1.000 | 0.0069 |
| Shapley set-CR | 0.0160 | 0.0308 | 0.964 | 0.905 | 0.0484 |
| Last-click | 0.0198 | 0.0551 | 0.893 | 0.810 | 0.1032 |
| Time-decay | 0.0216 | 0.0558 | 0.893 | 0.810 | 0.0282 |
| Shapley-Zhao (counts) | 0.0223 | 0.0555 | 0.893 | 0.810 | 0.0516 |
| Position-based | 0.0223 | 0.0568 | 0.893 | 0.810 | 0.0589 |
| Linear | 0.0230 | 0.0572 | 0.893 | 0.810 | 0.0489 |
| Markov removal | 0.0236 | 0.0644 | 0.893 | 0.810 | 0.0532 |
| First-click | 0.0242 | 0.0604 | 0.893 | 0.810 | 0.1696 |

**ASCA no-A 90% intervals:**

| channel | interval | truth |
|---|---|---|
| paid_search | [0.241, 0.270] | 0.264 |
| social | [0.174, 0.204] | 0.186 |
| display | [0.091, 0.136] | 0.112 |
| email | [0.114, 0.136] | 0.131 |
| organic_search | [0.137, 0.160] | 0.143 |
| direct | [0.096, 0.116] | 0.111 |
| referral | [0.047, 0.069] | 0.054 |

All 7 intervals contain the truth. The bootstrap intervals for Markov cover only 2 of 7 channels, and set-CR covers 3 of 7. Those bands are tight around biased values.

**Fitted shape and effects:** h = 179h (truth 168h), κ = 0.81 (truth 1). The estimated weights are smaller than the true ones but in the same order. That is because κ and the w_c trade off against each other.

**Incremental conversions:** ASCA implies 2,166 incremental conversions. The truth on the subsample, Σ(conv_prob − σ(−3.2)), is 2,113, so ASCA is 2.5% high.

**Agent node on this data:** the benchmark has no agent node. The latent-A model learns d = 0.01 with posterior sd 2.45, which is just the prior. It correctly collapses to the no-A model, and its shares are identical. But the Laplace bands for that unidentified model are uninformative: they are wide and include negative values.

## Results: agentic simulation (truth = interventional Shapley under the true SCM)
| model | MAE | MaxAE | Spearman | Kendall | credit to agent_checkout (truth 0) | credit to dark AI channel (truth 0.202) |
|---|---|---|---|---|---|---|
| ASCA v0 (A + proxy + Z) | 0.0060 | 0.0151 | 1.000 | 1.000 | 0 | 0.209 [0.186, 0.225] |
| ASCA, no proxy | 0.0055 | 0.0198 | 1.000 | 1.000 | 0 | 0.202 |
| ASCA, no A | 0.0071 | 0.0187 | 1.000 | 1.000 | 0 | 0.217 |
| ASCA, no Z adjustment | 0.0274 | 0.0665 | 0.867 | 0.778 | 0 | 0.136 |
| First-click | 0.0502 | 0.2023 | 0.527 | 0.479 | 0 | 0 |
| Linear | 0.0538 | 0.2023 | 0.450 | 0.389 | 0.089 | 0 |
| Markov removal | 0.0617 | 0.2023 | 0.300 | 0.222 | 0.138 | 0 |
| Shapley set-CR | 0.0723 | 0.2558 | −0.033 | 0.056 | 0.256 | 0 |
| ASCA misuse (checkout as an exposure) | 0.0831 | 0.3698 | 0.100 | 0.167 | 0.370 | 0.044 |
| Last-click | 0.0929 | 0.3754 | −0.017 | 0.056 | 0.375 | 0 |

The full table is in `results/simulation_scores.md`. 7 of 9 ASCA v0 intervals cover the truth: paid_search and social fall just outside.

Fraction of each channel's credit that flows through A, ASCA v0 vs truth:

| channel | ASCA v0 | truth |
|---|---|---|
| ai_answer_content | 0.88 | 1.00 |
| referral | 0.61 | 0.64 |
| organic_search | 0.38 | 0.51 |
| social | 0.41 | 0.27 |
| paid_search | 0.15 | 0.13 |
| display | 0.05 | 0.07 |
| direct | 0.07 | 0.07 |
| email | 0.14 | 0.07 |

Without the proxy, this decomposition breaks down: some fractions come out negative and others above 1.

## Key findings
1. **Structural interventional credit wins on the benchmark.** ASCA's MAE is 0.0057, against 0.016 for the best baseline (set-CR) and 0.020–0.024 for the heuristics. It also gets the channel ranking exactly right, and its per-journey MAE is 4–25× lower.
   All path heuristics over-credit display and referral, which appear often but have low weight, and under-credit paid_search by 3–6 points. They are rewarding frequency in paths, not causal effect.
2. **Path models misattribute systematically in an agentic world.**
   * The observable but non-causal `agent_checkout` step gets 37.5% of credit from last-click, 26% from set-CR, 18% from time-decay and 14% from Markov.
   * The dark AI-answer channel, worth about 20% of true credit, gets 0 from every path model.
   * Rank correlation with the truth falls to about 0 for last-click and set-CR.
3. **Confounder adjustment and correct causal roles matter more than the agent node for total shares.**
   * Dropping the Z adjustment raises MAE about 4.5× (0.027), inflating email and direct, the channels driven by retargeting and brand.
   * Treating agent_checkout as an exposure (a post-treatment error) gives MAE 0.083, as bad as last-click.
4. **The latent agent node is mainly needed for path-specific (mediation) credit.**
   * Total shares come out about the same with or without A (MAE 0.0055–0.0071), because the marginal P(Y|X) is close to logistic either way.
   * Separating the X→A→Y path from X→Y requires an observed proxy for A (here, the AI-executed checkout flag). With the proxy, the mediated fractions are recovered roughly. Without it, they are not identified.
5. **Uncertainty is calibrated where the model is identified.**
   * ASCA's intervals cover the truth for 7/7 channels on the benchmark and 7/9 in the simulation. The bootstrap bands for Markov and set-CR are tight but miss the truth.
   * Laplace bands are unreliable for unidentified latent parameters, which is the benchmark latent-A case.

## Assumptions and honest limitations
* **The benchmark favors ASCA by construction.** Its truth is Shapley over a logistic value function on decayed, saturated exposure. ASCA's parametric family (exponential adstock, Hill saturation, logistic link) contains that DGP, so ASCA is close to correctly specified there. h and κ were learned, not fixed, but the *family* matches. Expect larger errors on real data. The next step is to test against misspecified DGPs.
* **The simulation is my own.** Its DGP shares ASCA's functional form, it has no unobserved confounding, and the agent proxy has no false positives. Those are optimistic assumptions.
* **Identification assumptions:**
  * No unobserved confounding given Z.
  * Exposures are well defined and can be intervened on.
  * The do(X_c = 0) contrast removes all of a channel's exposure.
  * Outcomes are evaluated at T_end = 720h, because the dataset has no conversion timestamps.
  * There is no interference between users.
* **Dark channels need an exposure log.** ASCA can credit "dark" channels only if their exposure is measured somehow: impression logs, assignment, or geo or flight variation. With no exposure data, no observational method can credit them. That case needs experiments or MMM.
* **The approximations are rough.** Laplace is a Gaussian approximation, not MCMC; I did not try PyMC or numpyro. The intervals reflect parameter uncertainty only, and the Laplace interval ignores the model-selection step. Gradients come from finite differences, which are slow: the simulation took about 27 minutes in total.
* **Some baselines rely on approximations or conventions:**
  * Markov's per-journey credit is an approximation (removal effect spread across the channels present).
  * Set-CR uses v(∅) = 0.
  * The time-decay half-life equals the DGP's half-life, which helps time-decay.

## Recommendations for ASCA v1
1. **Calibrate totals:** add incrementality and geo-lift or MMM calibration as a likelihood term, or a prior on the total incremental lift. That pins the totals when unobserved confounding is present.
2. **Use better inference:** move to a proper posterior with NUTS (numpyro or PyMC) and analytic gradients or JAX. Use non-centered priors and constrain d ≥ 0 and the effects ≥ 0, which fixes the unidentified latent-A bands.
3. **Model agents more richly:** use a multi-state agent node (Human-Owned / AI-Assisted / AI-Executed, following the IAB taxonomy), measured by several noisy proxies: agent user-agents, LLM referrers, checkout APIs. Report path-specific credit as standard output.
4. **Add a sequence layer:** use per-channel half-lives and saturation, and a hazard model in continuous time when conversion timestamps exist. That lets touches after conversion be excluded properly.
5. **Stress-test:** misspecified DGPs (interactions, carry-over, unobserved confounding, proxy false positives), plus a realism run on Criteo for scale and stability (no truth there). Report sensitivity bounds such as E-values or Rosenbaum-style bounds on the unobserved confounding.

## Fetching the benchmark tables

`journeys.parquet` and `touchpoints.parquet` are not in git (see `.gitignore`). On the hub they are Git LFS files at the **repository root** of [`lucianfialho/synthetic-attribution-benchmark`](https://huggingface.co/datasets/lucianfialho/synthetic-attribution-benchmark), revision `main` (commit `293a1054afa9ab1abefb21ab2ff14914d7bb52f0`). There is no `data/` directory on the hub.

| Hub path | Size | In this repo |
|---|---|---|
| `journeys.parquet` | 5,472,065 bytes | download to `data/journeys.parquet` |
| `touchpoints.parquet` | 12,772,646 bytes | download to `data/touchpoints.parquet` |
| `ground_truth.json` | 1,136 bytes | committed as `data/ground_truth.json` |

The same commit also has PNGs, notebooks, and `dda_comparison_results.json`. This project does not use those. The hub tags the repo `format:imagefolder` because of the PNGs, so `datasets.load_dataset` does not return the journey tables. `download_data.py` uses `huggingface_hub.hf_hub_download` and writes the two parquet files into `data/`.

```bash
python download_data.py
```

## Dataset attribution

The benchmark tables and `data/ground_truth.json` are from the Hugging Face dataset [`lucianfialho/synthetic-attribution-benchmark`](https://huggingface.co/datasets/lucianfialho/synthetic-attribution-benchmark) by the Hugging Face user `lucianfialho`. The dataset card (`data/README.md` in this repo) states the license: synthetic data, no PII, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Files
* `asca/common.py`: data structures, exact Shapley matrix, metrics.
* `asca/baselines.py`: baseline models.
* `asca/model.py`: ASCA SCM, Laplace inference, interventional Shapley credit.
* `run_benchmark.py`, `run_simulation.py`, `make_charts.py`
* `download_data.py`: fetch `journeys.parquet` and `touchpoints.parquet` into `data/`.
* `requirements.txt`: numpy, pandas, scipy, pyarrow, matplotlib, tabulate, huggingface_hub. Nothing in this repo imports scikit-learn.
* `results/`:
  * Benchmark: `benchmark_scores.{csv,md}`, `benchmark_shares.csv`, `benchmark_intervals.csv`, `benchmark_asca_fit.json`
  * Simulation: `simulation_scores.{csv,md}`, `simulation_shares.csv`, `simulation_intervals.csv`, `simulation_mediated_fraction.csv`, `simulation_fit.json`
  * Chart: `shares_vs_truth.png`

Reproduce: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && python download_data.py && python run_benchmark.py 100000 && python run_simulation.py && python make_charts.py`
