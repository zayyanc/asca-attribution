"""Estimated vs true channel shares per model (benchmark + simulation)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def panel(ax, shares_csv, truth_row, intervals_csv, title, asca_key):
    S = pd.read_csv(shares_csv, index_col=0)
    truth = S.loc[truth_row]; models = [m for m in S.index if not m.startswith(("Ground truth", "TRUE"))]
    ch = list(S.columns); x = np.arange(len(ch)); w = 0.8 / (len(models) + 1)
    cmap = plt.get_cmap("tab20")
    for i, m in enumerate(models):
        is_asca = m.startswith("ASCA")
        ax.bar(x + i * w, S.loc[m], w, label=m, color=cmap(i), edgecolor="black" if is_asca else "none", lw=0.6)
    ax.scatter(x + 0.4 - w / 2, truth, marker="_", s=900, color="black", lw=2.5, zorder=5, label=truth_row)
    if intervals_csv:
        I = pd.read_csv(intervals_csv, index_col=0)
        col = [c for c in I.columns if c.startswith(asca_key)][0]
        lo = I[col].str.extract(r"\[(-?[\d.]+),")[0].astype(float); hi = I[col].str.extract(r", (-?[\d.]+)\]")[0].astype(float)
        k = models.index([m for m in models if m.startswith(asca_key)][0])
        est = S.loc[models[k]]
        ax.errorbar(x + k * w, est, yerr=[est - lo.reindex(ch).values, hi.reindex(ch).values - est], fmt="none", ecolor="k", capsize=2, lw=1)
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(ch, rotation=25, ha="right"); ax.set_ylabel("credit share")
    ax.set_title(title); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=6.5, ncol=2)

fig, axes = plt.subplots(2, 1, figsize=(15, 11))
panel(axes[0], "results/benchmark_shares.csv", "Ground truth (500k)", "results/benchmark_intervals.csv",
      "Benchmark (lucianfialho/synthetic-attribution-benchmark, 100k stratified subsample): estimated vs true shares (error bar: 90% interval of ASCA no-A; latent-A Laplace bands are uninformative here, see README)",
      "ASCA v0 ablation: no A")
panel(axes[1], "results/simulation_shares.csv", "TRUE (interventional Shapley)", "results/simulation_intervals.csv",
      "Agentic simulation (dark AI-answer channel, latent agent node, non-causal agent_checkout): estimated vs true", "ASCA v0")
plt.tight_layout(); plt.savefig("results/shares_vs_truth.png", dpi=140)
print("saved")
