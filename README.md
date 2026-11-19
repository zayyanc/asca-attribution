# ASCA v0: Agentic Structural Causal Attribution

ASCA is a marketing attribution model for a world in which AI assistants influence, and sometimes complete, the purchase. This note has two parts. The first is for marketers and executives. The second is the technical reference: the model, the tests, and how to reproduce them.

Every number in the technical section comes from `run_benchmark.py` and `run_simulation.py`. The raw output is in `results/`.

## Contents

**For marketers and executives**

- [The problem](#the-problem)
- [What ASCA does differently](#what-asca-does-differently)
- [What the first results showed](#what-the-first-results-showed)
- [An honest caveat](#an-honest-caveat)
- [What comes next](#what-comes-next)

**Technical reference**

- [The model](#the-model)
- [How credit is assigned](#how-credit-is-assigned)
- [Baselines](#baselines)
- [Data](#data)
- [Download and reproduce](#download-and-reproduce)
- [Results](#results)
- [Assumptions, limitations, and v1](#assumptions-limitations-and-v1)
- [Dataset attribution](#dataset-attribution)

---

## For marketers and executives

### The problem

A customer used to leave a trail of clicks: an ad, a search, an email, a visit. The last click before the sale was easy to see, so last-click attribution handed that sale to whatever showed up last.

That trail is breaking. AI assistants and shopping agents now recommend products, compare them, and sometimes check out on the customer's behalf. Those steps often leave no click. The assistant can be the reason the sale happened, while the click log only shows a final checkout, or a channel that merely happened to be last. Last-click then credits the last visible step. The thing that actually caused the sale gets nothing.

### What ASCA does differently

Standard methods count touches along the path and give the sale to a position in that path: the last touch, the first touch, or a share of every touch.

ASCA asks a different question of each channel: **what would have happened to this sale if this channel had not been there?** Credit goes to the channels that change the outcome. A step that appears in the log but does not change the sale, including an AI checkout that only records a decision already made, gets no credit. A channel that never produces a click can still receive credit when there is a record that it was shown, for example an impression inside an AI answer.

### What the first results showed

Two tests, both with a known right answer.

**On a public benchmark of made-up shopping journeys, ASCA was the most accurate method, and it ranked the channels in exactly the right order.** The next-best method was clearly behind. The usual rules, including last-click, were further behind, and they did not get the ranking right. They gave too much credit to channels that show up often but matter less (display and referral), and they shorted paid search by 3 to 6 percentage points. Sale by sale, ASCA's split of credit was 4 to 25 times closer to the truth than those other methods. The uncertainty range ASCA reported contained the true credit for every channel. Ranges from the two usual methods we checked the same way were tight, and they missed the truth on most channels.

**On a separate test where an AI assistant sometimes takes part in the purchase, ASCA gave no credit to a checkout step that did not cause the sale, and it did credit an invisible AI-answer channel.** That invisible channel is worth about 20% of the true credit. Last-click gave it nothing, and gave the non-causal checkout step 37.5% of the credit. Other touch-counting rules did the same kind of thing. A method that scores which combination of channels appeared put about 26% on that checkout step, a method that favors recent touches put about 18% there, and a method that follows the usual order of steps put about 14% there. None of them credited the invisible channel.

| | Last-click | ASCA |
|---|---|---|
| Credit to the checkout step that did not cause the sale | 37.5% | none |
| Credit to the invisible AI-answer channel (true share about 20%) | none | credited, close to the true share |

### An honest caveat

The public benchmark favors ASCA's shape. Its right answer was built from the same kind of formula ASCA assumes: an ad's effect fades with time, extra exposures matter less once someone has already seen a lot, and the chance of buying follows that pattern. ASCA had to learn the fade rate and how quickly extra exposures wear out, but the family of the formula matches the test. Errors on real data should be larger. The next step is to test cases where that formula is wrong.

The assistant test was built by us. It uses that same formula, it has no hidden influence we failed to measure, and the checkout flag never turns on by mistake. Those are optimistic conditions.

### What comes next

1. Tie the overall totals to lift experiments and marketing-mix results, so the grand total stays honest when something unmeasured is pushing both exposure and sales.
2. Replace today's rough uncertainty ranges with a fuller fit, and stop those ranges from blowing up when the data cannot tell two explanations apart.
3. Describe the assistant in more than one state: a person shopping, a person helped by AI, or the AI completing the purchase. Use several noisy clues, and always say how much of a channel's credit ran through the assistant.
4. Let each channel fade on its own, and, when the time of the sale is known, ignore touches that happen after it.
5. Stress-test the model when the formula is wrong, and run it on a large real dataset (Criteo) where nobody knows the right answer, to see whether it stays stable.

---

## Technical reference

### The model

ASCA v0 (`asca/model.py`) is a structural causal model per journey $j$ and channel $c$. The pieces are:

- **Exposures** $X$: the measured marketing contacts, including a dark channel when its impressions are logged.
- **Latent agent node** $A$: whether an assistant mediated the journey. $A$ is not observed.
- **Confounders / segment** $Z$: observed pre-treatment covariates (the backdoor set). In the simulation, $Z$ is a high-intent segment.
- **Outcome** $Y$: conversion.
- **Proxy** $M$ (optional): an observed AI-executed checkout. $M$ is a descendant of $A$ and $Y$. It is never an exposure.

$$
\begin{aligned}
\operatorname{logit} P(A=1 \mid X, Z) &= a_0 + g \cdot s + z_A \cdot Z \\
\operatorname{logit} P(Y=1 \mid X, A, Z) &= b_0 + b \cdot s + d \cdot A + z_Y \cdot Z
\end{aligned}
$$

$s$ is saturated exposure, defined below. The benchmark has no agent node. The simulation does.

#### Exposure decay and saturation

Half-life $h$ and saturation scale $\kappa$ are free parameters.

$$
E_{jc}(h) = \sum_{\text{touches of } c} \exp\!\left(-\frac{\ln 2}{h}\,(T_{\mathrm{end}} - t)\right)
$$

$$
s_{jc} = \frac{E_{jc}}{\kappa + E_{jc}}
$$

Outcomes are evaluated at $T_{\mathrm{end}} = 720\,\mathrm{h}$, because the benchmark has no conversion timestamps. The benchmark's own decay uses a fixed half-life of $168\,\mathrm{h}$ inside the same window. See [Data](#data).

#### Likelihood, with $A$ marginalized

$A$ is latent and is marginalized out of the likelihood. With $\pi = P(A=1 \mid X, Z)$ and $\eta = b_0 + b \cdot s + z_Y \cdot Z$,

$$
P(Y=1 \mid X, Z) = (1-\pi)\,\sigma(\eta) + \pi\,\sigma(\eta + d)
$$

and $P(Y=0 \mid X, Z)$ is the complementary mixture. If the agent node is turned off, the likelihood is ordinary logistic regression of $Y$ on $s$ and $Z$.

#### The checkout flag is a descendant, not an exposure

The optional proxy is

$$
M \mid Y=1,\, A \sim \mathrm{Bernoulli}(q \cdot A)
$$

$M$ is the observed AI-executed checkout. Because it is a descendant of $A$ and $Y$, conditioning on it as if it were a marketing exposure is a post-treatment error. That misuse is one of the reported ablations: it is about as bad as last-click. With the proxy in the likelihood, a conversion with $M=1$ is attributed only to paths with $A=1$, since $P(M=1 \mid A=0) = 0$ under this specification.

#### Inference

MAP with weakly informative Gaussian priors: $\mathcal{N}(0, 2^2)$ on effects, $\mathcal{N}(0, 5^2)$ on intercepts, $\log h \sim \mathcal{N}(\log 100,\, 2^2)$, and $\log \kappa \sim \mathcal{N}(0,\, 2^2)$.

Uncertainty is a Laplace approximation. The Hessian is finite-differenced. Reported intervals use 200 posterior draws. Laplace is a Gaussian approximation, not MCMC. PyMC and numpyro were not used. The intervals reflect parameter uncertainty only, and they ignore the model-selection step.

### How credit is assigned

#### Interventional credit

For each converting journey, the value of a coalition $S$ of channels is the expected outcome after intervening to keep that journey's exposures in $S$ and to set every other channel to zero, still conditioning on the journey's $Z$:

$$
v_j(S) = \mathbb{E}\!\left[Y \mid do(X_S = x_{jS},\, X_{-S} = 0),\, Z_j\right]
$$

$\varphi_j$ is the exact Shapley value of $v_j$ over all $2^K$ coalitions. Rows are normalized to sum to 1 and then averaged across converting journeys. That is the same convention as the benchmark's `aggregate_true_contribution`: each converting journey counts equally.

Incremental conversions are

$$
\sum_j \left[v_j(\mathrm{all}) - v_j(\emptyset)\right]
$$

taken over all journeys, not only converters, and allocated by the un-normalized Shapley values.

#### Path-specific credit through the agent

The fraction of a channel's credit that flows through $A$ is

$$
1 - \frac{\varphi^{\mathrm{dir}}}{\varphi^{\mathrm{tot}}}
$$

where $\varphi^{\mathrm{dir}}$ is the Shapley value with the $A$-equation held at $X = 0$ (only the direct $X \to Y$ path is active). Separating $X \to A \to Y$ from $X \to Y$ needs an observed proxy for $A$. With the proxy, the mediated fractions are recovered roughly. Without it, they are not identified: some fractions come out negative and others above 1.

Total shares are a different question. They come out about the same with or without $A$ (MAE 0.0055–0.0071 on the simulation), because the marginal $P(Y \mid X)$ is close to logistic either way. The latent agent node is mainly needed for path-specific credit.

### Baselines

Every baseline produces credit per converting journey. Each journey's credit is normalized to 1, then averaged, matching the ground-truth convention.

- **Last-click, first-click, linear.**
- **Time-decay:** half-life $168\,\mathrm{h}$, measured back from $t = 720\,\mathrm{h}$. That half-life equals the data-generating half-life, which helps time-decay.
- **Position-based:** 40/20/40.
- **Markov:** a first-order chain with removal effect. The aggregate share is the normalized removal effect. Per journey, the removal effect is spread across the channels present. That per-journey split is an approximation.
- **Shapley-Zhao (counts):** $v(S) = \sum_{T \subseteq S} \mathrm{conv}(T)$. This reduces to an equal split across the distinct channels in a journey.
- **Shapley set-CR:** for each converting journey, exact Shapley with $v(S)$ equal to the empirical conversion rate of journeys whose exact channel set is $S$. Here $v(\emptyset) = 0$, unseen sets count as 0, and negative values are clipped.

### Data

**Public benchmark.** Hugging Face [`lucianfialho/synthetic-attribution-benchmark`](https://huggingface.co/datasets/lucianfialho/synthetic-attribution-benchmark) v2 (CC BY 4.0), downloaded into `data/`. The full set has 500,000 journeys, 1,349,161 touchpoints, and 29,938 conversions (5.99%).

The scored subsample is 100,000 journeys stratified on `converted`, seed 7. That gives 5,988 conversions (5.988% vs 5.988% in the full set) and 269,817 touchpoints. `results/benchmark_scores.md` records the same subsample as `conv_rate_sub` 0.05988 and `conv_rate_full` 0.059876.

The data-generating process, checked in code, is

$$
p = \sigma\!\left(-3.2 + \sum_c w_c \cdot s(D_c)\right), \quad s(x) = \frac{x}{1+x}, \quad D_c = \sum \exp\!\left(-\frac{\ln 2}{168\,\mathrm{h}}\,(720 - t)\right)
$$

This reproduces `conv_prob` to 6e-7. For every converting journey, $\sum_c \mathrm{gt\_shapley}_c = \mathrm{conv\_prob} - \sigma(-3.2)$, with error 5e-7.

Channel weights: paid_search 1.1, social 0.7, display 0.35, email 0.9, organic_search 0.8, direct 1.0, referral 0.5.

Truth for scoring is `aggregate_true_contribution` from `ground_truth.json`, built from per-journey normalized Shapley values with each conversion weighted equally. The same quantity is also recomputed on the subsample.

`dda_comparison_results.json` on the hub has different "gt_shapley" numbers. They look like the v1 aggregation convention, so they were not used.

**Agentic simulation.** `run_simulation.py` builds its own journeys with an explicit agent node. It has 200,000 journeys and 17,907 conversions (8.95%). 50.1% of conversions are agent-mediated ($A = 1$). 37.5% of conversions show an observed `agent_checkout`. The file header in `results/simulation_scores.md` is `N=200000`, conversions `17907` (`0.0895`), agent-mediated conversions `0.501`, observed `agent_checkout` among conversions `0.375`.

Truth on the simulation is interventional Shapley under the true structural model. The truth for `agent_checkout` is 0. The dark channel `ai_answer_content` is logged impressions of brand content inside LLM answers: ASCA sees it as an exposure, and the path models do not.

### Download and reproduce

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

`requirements.txt` is numpy, pandas, scipy, pyarrow, matplotlib, tabulate, and huggingface_hub. Nothing in this repo imports scikit-learn.

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && python download_data.py && python run_benchmark.py 100000 && python run_simulation.py && python make_charts.py
```

`run_benchmark.py` reads the parquet files. `run_simulation.py` and `make_charts.py` do not. The simulation took about 27 minutes in total. Gradients are finite differences, which is why.

### Results

#### Benchmark (100k subsample, scored against the 500k truth)

| model | MAE | MaxAE | Spearman | Kendall | MAE vs subsample truth | per-journey MAE | 90% coverage |
|---|---|---|---|---|---|---|---|
| ASCA v0 ablation: no A | 0.0057 | 0.0098 | 1.0000 | 1.0000 | 0.0047 | 0.0069 | 1.0000 |
| ASCA v0 (latent agent A) | 0.0057 | 0.0098 | 1.0000 | 1.0000 | 0.0047 | 0.0069 | 1.0000 |
| Shapley set-CR | 0.0160 | 0.0308 | 0.9643 | 0.9048 | 0.0153 | 0.0484 | 0.4286 |
| Last-click | 0.0198 | 0.0551 | 0.8929 | 0.8095 | 0.0194 | 0.1032 | |
| Time-decay (h=168h) | 0.0216 | 0.0558 | 0.8929 | 0.8095 | 0.0211 | 0.0282 | |
| Shapley-Zhao (counts) | 0.0223 | 0.0555 | 0.8929 | 0.8095 | 0.0218 | 0.0516 | |
| Position-based 40/20/40 | 0.0223 | 0.0568 | 0.8929 | 0.8095 | 0.0218 | 0.0589 | |
| Linear | 0.0230 | 0.0572 | 0.8929 | 0.8095 | 0.0226 | 0.0489 | |
| Markov removal | 0.0236 | 0.0644 | 0.8929 | 0.8095 | 0.0233 | 0.0532 | 0.2857 |
| First-click | 0.0242 | 0.0604 | 0.8929 | 0.8095 | 0.0237 | 0.1696 | |

Blank coverage cells were not computed. ASCA's MAE is 0.0057, against 0.016 for the best baseline (set-CR) and 0.020–0.024 for the heuristics. Spearman 1.000 means the channel ranking is exactly right. Per-journey MAE is 4–25× lower than the baselines. The previous summary rounded Spearman and Kendall to three decimals: 0.964 and 0.905 for set-CR, and 0.893 and 0.810 for every path heuristic. The table above is the four-decimal printout in `results/benchmark_scores.md`.

**ASCA no-A 90% intervals** (Laplace posterior). These are the bands plotted on the benchmark chart.

| channel | interval | truth |
|---|---|---|
| paid_search | [0.241, 0.270] | 0.264 |
| social | [0.174, 0.204] | 0.186 |
| display | [0.091, 0.136] | 0.112 |
| email | [0.114, 0.136] | 0.131 |
| organic_search | [0.137, 0.160] | 0.143 |
| direct | [0.096, 0.116] | 0.111 |
| referral | [0.047, 0.069] | 0.054 |

All 7 intervals contain the truth. The bootstrap intervals for Markov cover only 2 of 7 channels, and set-CR covers 3 of 7. Those bands are tight around biased values. The latent-A Laplace bands on this benchmark are a different object: coverage in the score table is 1.0000 only because the intervals are huge. They are uninformative and include negative values. From `results/benchmark_intervals.csv`:

| channel | Markov removal | Shapley set-CR | ASCA v0 (latent agent A) | ASCA v0 ablation: no A |
|---|---|---|---|---|
| paid_search | [0.196, 0.202] | [0.206, 0.248] | [0.188, 0.564] | [0.241, 0.270] |
| social | [0.181, 0.188] | [0.170, 0.214] | [-0.030, 0.361] | [0.174, 0.204] |
| display | [0.169, 0.174] | [0.126, 0.158] | [-0.541, 0.216] | [0.091, 0.136] |
| email | [0.119, 0.125] | [0.093, 0.126] | [0.099, 0.262] | [0.114, 0.136] |
| organic_search | [0.141, 0.146] | [0.129, 0.158] | [0.112, 0.332] | [0.137, 0.160] |
| direct | [0.101, 0.105] | [0.093, 0.120] | [0.085, 0.268] | [0.096, 0.116] |
| referral | [0.074, 0.079] | [0.063, 0.090] | [-0.188, 0.118] | [0.047, 0.069] |

**Fitted shape.** $h = 179\,\mathrm{h}$ (truth $168\,\mathrm{h}$), $\kappa = 0.81$ (truth 1). The estimated weights are smaller than the true ones but in the same order, because $\kappa$ and the $w_c$ trade off against each other.

**Incremental conversions.** ASCA implies 2,166 incremental conversions. The truth on the subsample, $\sum(\mathrm{conv\_prob} - \sigma(-3.2))$, is 2,113, so ASCA is 2.5% high.

**Agent node on this data.** The benchmark has no agent node. The latent-A model learns $d = 0.01$ with posterior sd 2.45, which is just the prior. It correctly collapses to the no-A model, and its shares are identical.

Channel shares (`results/benchmark_shares.csv`):

| model | paid_search | social | display | email | organic_search | direct | referral |
|---|---|---|---|---|---|---|---|
| Last-click | 0.2248 | 0.1877 | 0.1672 | 0.1176 | 0.1408 | 0.0959 | 0.0661 |
| First-click | 0.2034 | 0.1951 | 0.1707 | 0.1207 | 0.1428 | 0.0970 | 0.0703 |
| Linear | 0.2145 | 0.1950 | 0.1692 | 0.1168 | 0.1399 | 0.0966 | 0.0679 |
| Time-decay (h=168h) | 0.2201 | 0.1935 | 0.1679 | 0.1165 | 0.1397 | 0.0965 | 0.0658 |
| Position-based 40/20/40 | 0.2139 | 0.1929 | 0.1688 | 0.1183 | 0.1416 | 0.0966 | 0.0680 |
| Shapley-Zhao (counts) | 0.2106 | 0.1907 | 0.1675 | 0.1192 | 0.1411 | 0.0996 | 0.0713 |
| Shapley set-CR | 0.2346 | 0.1905 | 0.1429 | 0.1105 | 0.1436 | 0.1046 | 0.0733 |
| Markov removal | 0.1994 | 0.1836 | 0.1718 | 0.1231 | 0.1429 | 0.1027 | 0.0765 |
| ASCA v0 (latent agent A) | 0.2540 | 0.1875 | 0.1195 | 0.1256 | 0.1492 | 0.1064 | 0.0578 |
| ASCA v0 ablation: no A | 0.2541 | 0.1875 | 0.1194 | 0.1256 | 0.1492 | 0.1064 | 0.0579 |
| Ground truth (500k) | 0.2638 | 0.1856 | 0.1120 | 0.1313 | 0.1425 | 0.1107 | 0.0540 |
| Ground truth (subsample) | 0.2617 | 0.1864 | 0.1119 | 0.1286 | 0.1444 | 0.1121 | 0.0549 |

All path heuristics over-credit display and referral, which appear often but have low weight, and under-credit paid_search by 3–6 points. They reward frequency in paths, not causal effect.

#### Agent simulation (truth = interventional Shapley under the true model)

| model | MAE | MaxAE | Spearman | Kendall | per-journey MAE | credit to agent_checkout (truth 0) | credit to dark AI channel (truth 0.202) | 90% coverage |
|---|---|---|---|---|---|---|---|---|
| ASCA ablation: A latent, no proxy | 0.0055 | 0.0198 | 1.0000 | 1.0000 | 0.0082 | 0.0000 | 0.2020 | 1.0000 |
| ASCA v0 (A + proxy + Z) | 0.0060 | 0.0151 | 1.0000 | 1.0000 | 0.0071 | 0.0000 | 0.2094 | 0.7778 |
| ASCA ablation: no A node | 0.0071 | 0.0187 | 1.0000 | 1.0000 | 0.0083 | 0.0000 | 0.2170 | 0.6667 |
| ASCA ablation: no confounder adjustment | 0.0274 | 0.0665 | 0.8667 | 0.7778 | 0.0351 | 0.0000 | 0.1358 | 0.3333 |
| First-click | 0.0502 | 0.2023 | 0.5272 | 0.4789 | 0.1627 | 0.0000 | 0.0000 | |
| Linear | 0.0538 | 0.2023 | 0.4500 | 0.3889 | 0.0787 | 0.0894 | 0.0000 | |
| Shapley-Zhao (counts) | 0.0576 | 0.2023 | 0.4500 | 0.3889 | 0.0840 | 0.1054 | 0.0000 | |
| Position-based 40/20/40 | 0.0591 | 0.2023 | 0.1167 | 0.1111 | 0.1029 | 0.1529 | 0.0000 | |
| Time-decay (h=168h) | 0.0611 | 0.2023 | 0.1167 | 0.1111 | 0.0806 | 0.1826 | 0.0000 | |
| Markov removal | 0.0617 | 0.2023 | 0.3000 | 0.2222 | 0.0820 | 0.1375 | 0.0000 | |
| Shapley set-CR | 0.0723 | 0.2558 | −0.0333 | 0.0556 | 0.1032 | 0.2558 | 0.0000 | |
| ASCA misuse (checkout as an exposure) | 0.0831 | 0.3698 | 0.1000 | 0.1667 | 0.1030 | 0.3698 | 0.0442 | 0.1111 |
| Last-click | 0.0929 | 0.3754 | −0.0167 | 0.0556 | 0.1575 | 0.3754 | 0.0000 | |

The model names in `results/simulation_scores.md` are slightly longer (`ASCA v0 (A latent + agent_checkout proxy, Z adjusted)`, and so on). The table above is that file's four-decimal printout. The previous summary rounded the rows it showed to three decimals: no-$Z$ Spearman 0.867 and Kendall 0.778; first-click 0.527 and 0.479; linear 0.450 and 0.389; Markov 0.300 and 0.222; set-CR −0.033 and 0.056; checkout-as-exposure 0.100 and 0.167; last-click −0.017 and 0.056. Checkout credit in that summary was 0.089 (linear), 0.138 (Markov), 0.256 (set-CR), 0.370 (checkout treated as an exposure), and 0.375 (last-click). Dark-channel credit was 0.209 [0.186, 0.225] for ASCA v0 against truth 0.202, 0.202 with no proxy, 0.217 with no $A$, 0.136 with no $Z$, and 0.044 when the checkout was treated as an exposure. The prose also stated those checkout shares as 37.5% last-click, 26% set-CR, 18% time-decay, and 14% Markov.

7 of 9 ASCA v0 intervals cover the truth: paid_search and social fall just outside. Rank correlation with the truth falls to about 0 for last-click and set-CR.

Dropping the $Z$ adjustment raises MAE about 4.5× (0.027), inflating email and direct, the channels driven by retargeting and brand. Treating `agent_checkout` as an exposure gives MAE 0.083, as bad as last-click (0.0929).

ASCA v0 90% intervals (`results/simulation_intervals.csv`):

| channel | interval |
|---|---|
| paid_search | [0.224, 0.250] |
| social | [0.107, 0.129] |
| display | [0.036, 0.072] |
| email | [0.124, 0.147] |
| organic_search | [0.097, 0.114] |
| direct | [0.083, 0.108] |
| referral | [0.039, 0.052] |
| ai_answer_content | [0.186, 0.225] |
| agent_checkout | [0.000, 0.000] |

The other variants' intervals are in the same file. The no-proxy dark-channel interval is [-0.181, 0.230], which is why that variant's coverage can be high while the mediation split is not identified.

Fraction of each channel's credit flowing through $A$. Rounded to two decimals, ASCA v0 vs truth:

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

Three decimals, including the ablations, from `results/simulation_scores.md`:

| channel | ASCA v0 | no proxy | no Z adjustment | truth |
|---|---|---|---|---|
| paid_search | 0.146 | 0.325 | 0.271 | 0.133 |
| social | 0.413 | −0.013 | 0.477 | 0.270 |
| display | 0.051 | −0.704 | 0.191 | 0.067 |
| email | 0.138 | 0.650 | 0.278 | 0.066 |
| organic_search | 0.381 | 0.536 | 0.454 | 0.512 |
| direct | 0.070 | 0.299 | 0.259 | 0.066 |
| referral | 0.606 | 0.868 | 0.675 | 0.644 |
| ai_answer_content | 0.878 | 0.978 | 1.080 | 1.000 |

Channel shares (`results/simulation_shares.csv`):

| model | paid_search | social | display | email | organic_search | direct | referral | ai_answer_content | agent_checkout |
|---|---|---|---|---|---|---|---|---|---|
| Last-click | 0.1485 | 0.0968 | 0.0869 | 0.1068 | 0.0698 | 0.0838 | 0.0319 | 0.0000 | 0.3754 |
| First-click | 0.2280 | 0.1562 | 0.1447 | 0.1714 | 0.1115 | 0.1328 | 0.0554 | 0.0000 | 0.0000 |
| Linear | 0.2123 | 0.1418 | 0.1278 | 0.1552 | 0.1031 | 0.1203 | 0.0500 | 0.0000 | 0.0894 |
| Time-decay (h=168h) | 0.1929 | 0.1268 | 0.1125 | 0.1399 | 0.0925 | 0.1083 | 0.0445 | 0.0000 | 0.1826 |
| Position-based 40/20/40 | 0.1966 | 0.1321 | 0.1202 | 0.1446 | 0.0951 | 0.1123 | 0.0461 | 0.0000 | 0.1529 |
| Shapley-Zhao (counts) | 0.1945 | 0.1417 | 0.1303 | 0.1469 | 0.1073 | 0.1196 | 0.0542 | 0.0000 | 0.1054 |
| Shapley set-CR | 0.1813 | 0.1062 | 0.0907 | 0.1390 | 0.0790 | 0.1128 | 0.0352 | 0.0000 | 0.2558 |
| Markov removal | 0.1762 | 0.1382 | 0.1287 | 0.1389 | 0.1073 | 0.1142 | 0.0591 | 0.0000 | 0.1375 |
| ASCA v0 (A + proxy + Z) | 0.2362 | 0.1186 | 0.0546 | 0.1344 | 0.1057 | 0.0955 | 0.0456 | 0.2094 | 0.0000 |
| ASCA, no proxy | 0.2362 | 0.1275 | 0.0640 | 0.1281 | 0.1039 | 0.0952 | 0.0433 | 0.2020 | 0.0000 |
| ASCA, no A | 0.2326 | 0.1193 | 0.0522 | 0.1324 | 0.1059 | 0.0938 | 0.0467 | 0.2170 | 0.0000 |
| ASCA, no Z adjustment | 0.2503 | 0.0979 | 0.0603 | 0.1927 | 0.0844 | 0.1430 | 0.0356 | 0.1358 | 0.0000 |
| ASCA misuse (checkout as an exposure) | 0.1831 | 0.0779 | 0.0484 | 0.1072 | 0.0691 | 0.0790 | 0.0213 | 0.0442 | 0.3698 |
| TRUE (interventional Shapley) | 0.2513 | 0.1297 | 0.0441 | 0.1353 | 0.1038 | 0.0934 | 0.0400 | 0.2023 | 0.0000 |

![Estimated versus true channel shares](results/shares_vs_truth.png)

#### What the tables add up to

1. **Structural interventional credit wins on the benchmark.** MAE 0.0057 versus 0.016 for set-CR and 0.020–0.024 for the heuristics, with the channel ranking exactly right and per-journey MAE 4–25× lower.
2. **Path models misattribute in the agentic test.** `agent_checkout` gets 37.5% from last-click, 26% from set-CR, 18% from time-decay, and 14% from Markov. The dark AI-answer channel, worth about 20% of true credit, gets 0 from every path model.
3. **Confounder adjustment and correct causal roles matter more than the agent node for total shares.** Dropping $Z$ raises MAE about 4.5× (0.027). Treating the checkout as an exposure gives MAE 0.083.
4. **The latent agent node is mainly for path-specific credit.** Total-share MAE is 0.0055–0.0071 with or without $A$. The mediation split needs the proxy.
5. **Uncertainty is calibrated where the model is identified.** Intervals cover the truth for 7/7 channels on the benchmark (the no-A model) and 7/9 in the simulation. Markov and set-CR bootstrap bands are tight and miss the truth. Laplace bands are unreliable for unidentified latent parameters, which is the benchmark latent-A case.

### Assumptions, limitations, and v1

- **The benchmark favors ASCA by construction.** Its truth is Shapley over a logistic value function on decayed, saturated exposure. ASCA's parametric family (exponential adstock, Hill saturation, logistic link) contains that data-generating process, so ASCA is close to correctly specified there. $h$ and $\kappa$ were learned, not fixed, but the family matches. Expect larger errors on real data. The next step is to test against misspecified processes.
- **The simulation is our own.** Its data-generating process shares ASCA's functional form, it has no unobserved confounding, and the agent proxy has no false positives. Those are optimistic assumptions.
- **Identification.** No unobserved confounding given $Z$. Exposures are well defined and can be intervened on. The $do(X_c = 0)$ contrast removes all of a channel's exposure. Outcomes are evaluated at $T_{\mathrm{end}} = 720\,\mathrm{h}$. There is no interference between users.
- **Dark channels need an exposure log.** ASCA can credit a dark channel only if its exposure is measured somehow: impression logs, assignment, or geo or flight variation. With no exposure data, no observational method can credit it. That case needs experiments or a marketing-mix model.
- **The approximations are rough.** Laplace is not MCMC. Finite-difference gradients made the simulation take about 27 minutes.
- **Baseline conventions.** Markov's per-journey credit spreads the removal effect across channels present. Set-CR uses $v(\emptyset) = 0$. Time-decay's half-life equals the data-generating half-life.

**v1 roadmap**

1. **Calibrate totals.** Add incrementality and geo-lift or marketing-mix calibration as a likelihood term, or a prior on the total incremental lift. That pins the totals when unobserved confounding is present.
2. **Use better inference.** Move to a proper posterior with NUTS (numpyro or PyMC) and analytic gradients or JAX. Use non-centered priors and constrain $d \ge 0$ and the effects $\ge 0$, which fixes the unidentified latent-A bands.
3. **Model agents more richly.** Use a multi-state agent node (Human-Owned / AI-Assisted / AI-Executed, following the IAB taxonomy), measured by several noisy proxies: agent user-agents, LLM referrers, checkout APIs. Report path-specific credit as standard output.
4. **Add a sequence layer.** Per-channel half-lives and saturation, and a hazard model in continuous time when conversion timestamps exist, so touches after conversion can be excluded.
5. **Stress-test.** Misspecified processes (interactions, carry-over, unobserved confounding, proxy false positives), plus a realism run on Criteo for scale and stability (no truth there). Report sensitivity bounds such as E-values or Rosenbaum-style bounds on the unobserved confounding.

### Dataset attribution

The benchmark tables and `data/ground_truth.json` are from the Hugging Face dataset [`lucianfialho/synthetic-attribution-benchmark`](https://huggingface.co/datasets/lucianfialho/synthetic-attribution-benchmark) by the Hugging Face user `lucianfialho`. The dataset card (`data/README.md` in this repo) states the license: synthetic data, no PII, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

### Files

- `asca/common.py`: data structures, exact Shapley matrix, metrics.
- `asca/baselines.py`: baseline models.
- `asca/model.py`: ASCA structural model, Laplace inference, interventional Shapley credit.
- `run_benchmark.py`, `run_simulation.py`, `make_charts.py`
- `download_data.py`: fetch `journeys.parquet` and `touchpoints.parquet` into `data/`.
- `requirements.txt`
- `results/`
  - Benchmark: `benchmark_scores.{csv,md}`, `benchmark_shares.csv`, `benchmark_intervals.csv`, `benchmark_asca_fit.json`, `benchmark_log.txt`
  - Simulation: `simulation_scores.{csv,md}`, `simulation_shares.csv`, `simulation_intervals.csv`, `simulation_mediated_fraction.csv`, `simulation_fit.json`
  - Chart: `shares_vs_truth.png`
