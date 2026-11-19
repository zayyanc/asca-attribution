---
license: cc-by-4.0
language:
- en
size_categories:
- 100K<n<1M
task_categories:
- tabular-classification
tags:
- marketing
- attribution
- multi-touch-attribution
- synthetic
- ground-truth
- shapley
- cro
pretty_name: Synthetic Multi-Touch Attribution Benchmark (with ground truth)
---

# Synthetic Multi-Touch Attribution Benchmark

A **synthetic** marketing dataset of multi-touch user journeys across interpretable
channels, with a **known data-generating process (DGP)** — so the **true per-channel
credit (exact Shapley value)** for every converting journey is included.

Real attribution datasets (e.g. Criteo) are hashed and have **no ground truth**, so you
cannot tell whether last-click / linear / time-decay / Markov / Shapley attribution is
*correct*. Here you can: compare any attribution heuristic against `gt_shapley_*`.

> Fully reproducible from the seed. DGP parameters in `ground_truth.json`.

## Files
- `touchpoints.parquet` — one row per touchpoint: `journey_id, step, channel, t_hours, cost`.
- `journeys.parquet` — one row per journey: `journey_id, n_touch, n_channels, converted, conv_prob`, and the ground-truth credit `gt_shapley_<channel>` (0 if the channel is absent or the journey did not convert).
- `ground_truth.json` — the DGP parameters (channel weights, base logit, half-life) and the aggregate true channel contribution. `aggregate_true_contribution` normalizes each converting journey's Shapley credit to sum to 1 before pooling across journeys (every converting journey counts equally) — the same convention every evaluated attribution method uses when it normalizes its own per-journey credit, so MAE against this field is an apples-to-apples comparison.

## Channels & true weights (DGP)
| channel | weight | mix_weight | cost |
|---|---|---|---|
| paid_search | 1.1 | 0.9 | 1.2 |
| social | 0.7 | 0.85 | 0.8 |
| display | 0.35 | 0.8 | 0.4 |
| email | 0.9 | 0.5 | 0.05 |
| organic_search | 0.8 | 0.6 | 0.0 |
| direct | 1.0 | 0.4 | 0.0 |
| referral | 0.5 | 0.3 | 0.1 |

`mix_weight` is a **categorical mixture weight**, not an independent arrival rate.
Each touchpoint in a journey is sampled from a single categorical draw over all channels,
with probabilities proportional to these weights — not from independent Poisson processes
per channel. Journeys that happen to contain a channel multiple times do so by repeated
draws, not by a separate channel-level process.

Base logit = -3.2; exposure half-life = 168h; journey window = 720h.
Conversion ~ Bernoulli(sigmoid(base + Σ_channel weight·saturate(decayed_exposure))).
Ground-truth credit = exact Shapley value of each channel under that value function.

## Suggested uses
- **Benchmark attribution models**: score last-click/first-click/linear/time-decay/Markov
  against `gt_shapley_*` (MAE on channel credit shares).
- **tabular-classification**: predict `converted` from journey features.
- Teaching material for CRO / marketing attribution.

## Changelog

**v2 (current)** — two breaking changes from v1:
- **Size: 50,000 → 500,000 journeys** (29,938 converting), matching the scale used
  throughout the accompanying paper. The DGP, seed (42), and channel parameters are
  unchanged, so v1 rows are a subset of what v2 regenerates at the same seed — v1 was
  simply generated with a smaller `--users` value than the paper always described.
- **`aggregate_true_contribution` convention fixed.** v1 computed this field by summing
  raw per-journey Shapley credit across all converting journeys and normalizing once at
  the end — implicitly weighting each journey by its total Shapley mass. v2 normalizes
  each converting journey's credit to sum to 1 *before* pooling, so every converting
  journey counts equally. This is the convention every evaluated attribution method
  (Zhao 2018, LOO, temporal removal) uses when it normalizes its own per-journey output,
  so v2's ground truth is directly comparable to method output in MAE; v1's was not.
  Per-touchpoint and per-journey `gt_shapley_*` values (the row-level ground truth) are
  unaffected — only the pre-aggregated summary in `ground_truth.json` changes.
- If you have code that reproduces MAE numbers against v1's `ground_truth.json`, re-run
  against v2 — the aggregate contribution values differ by up to ~0.04 per channel
  between the two conventions.

## License
Synthetic data, no PII. CC BY 4.0.
