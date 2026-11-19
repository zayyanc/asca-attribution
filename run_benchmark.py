"""Score baselines + ASCA v0 on lucianfialho/synthetic-attribution-benchmark (stratified subsample)."""
import json, time, sys
import numpy as np, pandas as pd
from asca.common import Paths, score, per_journey_mae, normalize_rows
from asca import baselines as B
from asca.model import ASCA

N_SUB = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
N_DRAWS, N_BOOT, SEED = 200, 30, 7
OUT = "results"
gt = json.load(open("data/ground_truth.json"))
CH = list(gt["dgp"]["channels"])
truth_full = gt["aggregate_true_contribution"]

j = pd.read_parquet("data/journeys.parquet"); tp = pd.read_parquet("data/touchpoints.parquet")
rng = np.random.default_rng(SEED)
conv, nonc = j[j.converted == 1], j[j.converted == 0]
n_c = int(round(N_SUB * len(conv) / len(j)))
sub = pd.concat([conv.sample(n_c, random_state=SEED), nonc.sample(N_SUB - n_c, random_state=SEED)]).sort_values("journey_id")
P = Paths.from_frames(tp, sub.journey_id.values, sub.converted.values, CH)
gt_rows = sub.loc[sub.converted == 1, [f"gt_shapley_{c}" for c in CH]].values
truth_sub = dict(zip(CH, normalize_rows(gt_rows).mean(0)))
info = dict(n_sub=P.n, n_conv=int(P.y.sum()), conv_rate_sub=float(P.y.mean()), conv_rate_full=float(j.converted.mean()),
            n_touch=len(P.t))
print(info)

models, rows_by = {}, {}
for name, fn in [("Last-click", B.last_click), ("First-click", B.first_click), ("Linear", B.linear),
                 ("Time-decay (h=168h)", B.time_decay), ("Position-based 40/20/40", B.position_based),
                 ("Shapley-Zhao (counts)", B.shapley_zhao_counts), ("Shapley set-CR", B.shapley_set_cr)]:
    r = fn(P); rows_by[name] = r; models[name] = dict(zip(CH, r.mean(0)))
mr, magg = B.markov_removal(P)
rows_by["Markov removal"] = mr; models["Markov removal"] = dict(zip(CH, magg))   # aggregate = removal-effect shares

# bootstrap intervals for the data-driven baselines
boot = {"Markov removal": [], "Shapley set-CR": []}
for b in range(N_BOOT):
    Pb = P.subset(rng.integers(0, P.n, P.n))
    boot["Markov removal"].append(B.markov_removal(Pb)[1]); boot["Shapley set-CR"].append(B.shapley_set_cr(Pb).mean(0))

# ---- ASCA v0 ----
asca_res = {}
for name, kw in [("ASCA v0 (latent agent A)", dict(latent_agent=True)), ("ASCA v0 ablation: no A", dict(latent_agent=False))]:
    t0 = time.time(); print("fitting", name)
    m = ASCA(len(CH), **kw).fit(P)
    th = m.draws(N_DRAWS, seed=SEED)
    shares = np.array([normalize_rows(m.credit(t, P)).mean(0) for t in th])
    phi_map = m.credit(m.theta, P)
    rows_by[name] = normalize_rows(phi_map); models[name] = dict(zip(CH, rows_by[name].mean(0)))
    boot[name] = shares
    inc = m.incremental(m.theta, P)
    asca_res[name] = dict(summary={k: [float(a), float(b)] for k, (a, b) in m.summary().items()},
                          incremental_conversions=dict(zip(CH, inc.tolist())), incremental_total=float(inc.sum()),
                          observed_conversions=int(P.y.sum()), fit_seconds=time.time() - t0,
                          est_half_life=float(np.exp(m.unpack(m.theta)["log_h"])),
                          est_kappa=float(np.exp(m.unpack(m.theta)["log_kappa"])))
    print(f"  done in {time.time()-t0:.0f}s; h={asca_res[name]['est_half_life']:.1f} kappa={asca_res[name]['est_kappa']:.3f}")

# true incremental conversions on the subsample (from conv_prob - sigmoid(base))
base_p = 1 / (1 + np.exp(3.2))
true_inc_total = float((sub.conv_prob - base_p).sum())

# ---- scoring ----
tab = []
for name in models:
    s_full = score(models[name], truth_full, CH); s_sub = score(models[name], truth_sub, CH)
    row = dict(model=name, MAE=s_full["MAE"], MaxAE=s_full["MaxAE"], Spearman=s_full["Spearman"], Kendall=s_full["Kendall"],
               MAE_vs_subsample_truth=s_sub["MAE"], per_journey_MAE=per_journey_mae(rows_by[name], gt_rows))
    tab.append(row)
tab = pd.DataFrame(tab).sort_values("MAE")
shares = pd.DataFrame(models).T[CH]; shares.loc["Ground truth (500k)"] = [truth_full[c] for c in CH]
shares.loc["Ground truth (subsample)"] = [truth_sub[c] for c in CH]
ints = {}
for name, arr in boot.items():
    arr = np.asarray(arr)
    ints[name] = {c: (float(np.quantile(arr[:, i], .05)), float(np.quantile(arr[:, i], .95))) for i, c in enumerate(CH)}
    cover = np.mean([lo <= truth_full[c] <= hi for c, (lo, hi) in ints[name].items()])
    tab.loc[tab.model == name, "90%_interval_coverage_of_truth"] = cover
tab.to_csv(f"{OUT}/benchmark_scores.csv", index=False); shares.to_csv(f"{OUT}/benchmark_shares.csv")
iv = pd.DataFrame({f"{n} [5%,95%]": {c: f"[{lo:.3f}, {hi:.3f}]" for c, (lo, hi) in d.items()} for n, d in ints.items()})
iv.to_csv(f"{OUT}/benchmark_intervals.csv")
with open(f"{OUT}/benchmark_scores.md", "w") as f:
    f.write(f"Subsample: {info}\n\n" + tab.to_markdown(index=False, floatfmt=".4f") + "\n\n### Channel shares\n\n"
            + shares.to_markdown(floatfmt=".4f") + "\n\n### 90% intervals (ASCA: Laplace posterior; Markov/set-CR: bootstrap)\n\n" + iv.to_markdown() + "\n")
json.dump(dict(info=info, asca=asca_res, true_incremental_total_subsample=true_inc_total), open(f"{OUT}/benchmark_asca_fit.json", "w"), indent=1)
print(tab.to_string()); print(shares.round(4).to_string()); print(iv.to_string())
print("true incremental total (subsample):", true_inc_total, {k: v["incremental_total"] for k, v in asca_res.items()})
