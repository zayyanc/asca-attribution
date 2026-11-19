"""Simulate the B2B world once, score ASCA-B2B and the path rules across tracking qualities.

Numbers written under results/b2b/ are the only figures the README should quote.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from b2b.baselines import baseline_shares
from b2b.common import QUALITIES, SCORE_CHANNELS, coverage, rank_scores, visibility
from b2b.model import draw_shares, fit_asca
from b2b.observe import observe
from b2b.simulate import simulate_world
from b2b.truth import ground_truth

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results" / "b2b"
DATA = ROOT / "data" / "b2b"

BASELINE_NAMES = [
    "lead_source", "last_touch", "w_shaped", "z_shaped", "linear",
    "time_decay", "markov", "set_cr",
]
BASELINE_LABELS = {
    "lead_source": "Lead source (first touch field)",
    "last_touch": "Last touch before opportunity / last touch",
    "w_shaped": "W-shaped",
    "z_shaped": "Z-shaped / full path",
    "linear": "Linear",
    "time_decay": "Time-decay",
    "markov": "Markov removal",
    "set_cr": "Shapley set-CR",
    "asca_full": "ASCA-B2B",
    "asca_no_z": "ASCA-B2B, no firmographics",
    "asca_touch": "ASCA-B2B, touch log only",
    "asca_misuse": "ASCA misuse (AI-sourced as a covariate)",
}


def _save_world(world, obs_good):
    DATA.mkdir(parents=True, exist_ok=True)
    n = world["n"]
    accounts = pd.DataFrame({
        "account_id": np.arange(n),
        "segment": np.array(["smb", "mid_market", "enterprise"])[world["segment"]],
        "icp": world["icp"],
        "intent": world["intent"],
        "existing_relationship": world["existing"],
        "acv": world["acv"],
        "n_contacts": world["n_contacts"],
        "opportunity": world["y"],
        "closed_won": world["w"],
        "agent": world["agent"],
        "iab_class": world["iab"],
        "ai_executed_demo": world["ai_executed"],
        "holdout_abm": world["hold_abm"],
        "holdout_llm": world["hold_llm"],
        "gold_panel": world["gold"],
        "mqa_day": world["mqa_day"],
        "sql_day": world["sql_day"],
        "opp_day": world["opp_day"],
        "close_day": world["close_day"],
        "s_sales": world["s_sales"],
    })
    accounts.to_parquet(DATA / "accounts.parquet", index=False)
    contacts = pd.DataFrame(world["contacts"], columns=["account_id", "contact_id", "role", "role_weight"])
    contacts.to_parquet(DATA / "contacts.parquet", index=False)
    t = world["touches"]
    touches = pd.DataFrame({
        "account_id": t["account_id"],
        "contact_id": t["contact_id"],
        "channel": np.array(world["channels"])[t["channel_idx"]],
        "day": t["day"],
        "role_weight": t["role_weight"],
        "post_decision": t["post_decision"],
    })
    touches.to_parquet(DATA / "touches_true.parquet", index=False)
    seen = obs_good["seen"]
    obs_touches = pd.DataFrame({
        "observed_account_id": obs_good["obs_account"][seen],
        "true_account_id": t["account_id"][seen],
        "channel": np.array(world["channels"])[t["channel_idx"][seen]],
        "day": t["day"][seen],
        "post_decision": t["post_decision"][seen],
    })
    obs_touches.to_parquet(DATA / "touches_observed_good.parquet", index=False)
    # Exposure matrices are regenerable; store the gold-panel calibration target only.
    pd.DataFrame(world["e"], columns=world["mkt"]).assign(account_id=np.arange(n)).to_parquet(
        DATA / "exposure_true.parquet", index=False)


def _boot(credit, acv, positive, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(acv)
    shares = np.zeros((n_boot, credit.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        w = np.where(positive[idx], acv[idx], 0.0)
        num = (credit[idx] * w[:, None]).sum(0)
        denom = num.sum()
        shares[b] = num / denom if denom > 0 else 0.0
    return np.quantile(shares, 0.05, axis=0), np.quantile(shares, 0.95, axis=0)


def _row(quality, model, outcome, pred, truth, lo=None, hi=None, identified=None, extra=None):
    stats = rank_scores(pred, truth)
    stats.update(quality=quality, model=model, outcome=outcome)
    if lo is not None:
        stats["coverage_90"] = coverage(lo, hi, truth)
        if identified is not None:
            mask = np.array(identified)
            stats["coverage_90_identified"] = coverage(lo[mask], hi[mask], truth[mask]) if mask.any() else float("nan")
            stats["n_identified"] = int(mask.sum())
    if extra:
        stats.update(extra)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--method", choices=["laplace", "nuts", "map"], default="nuts")
    parser.add_argument("--draws", type=int, default=60)
    parser.add_argument("--boot", type=int, default=40)
    parser.add_argument("--qualities", default="near_perfect,good,degraded,heavy")
    parser.add_argument("--nuts-warmup", type=int, default=150)
    parser.add_argument("--nuts-samples", type=int, default=80)
    args = parser.parse_args()

    print(f"simulating n={args.n} seed={args.seed}", flush=True)
    world = simulate_world(args.n, args.seed)
    truth = ground_truth(world)
    if abs(truth["sales_pipeline_share"]) > 1e-8:
        raise SystemExit(f"sales pipeline share should be 0, got {truth['sales_pipeline_share']}")
    print(
        f"opp rate={world['y'].mean():.3f} win|opp={(world['w'][world['y']].mean() if world['y'].any() else 0):.3f} "
        f"agent|opp={world['agent'][world['y']].mean():.3f} "
        f"ai_executed|opp={world['ai_executed'][world['y']].mean():.3f} "
        f"mean contacts={world['n_contacts'].mean():.2f} mean acv={world['acv'].mean():.0f}",
        flush=True,
    )
    print("pipeline truth", {k: round(truth["pipeline"][k], 4) for k in SCORE_CHANNELS}, flush=True)
    print("revenue truth", {k: round(truth["revenue"][k], 4) for k in SCORE_CHANNELS}, flush=True)

    qualities = [q.strip() for q in args.qualities.split(",") if q.strip()]
    obs_by_q = {}
    for name in qualities:
        print(f"observing {name}", flush=True)
        obs_by_q[name] = observe(world, QUALITIES[name], seed=args.seed + 100 + int(QUALITIES[name] * 1000))
    if "good" in obs_by_q:
        _save_world(world, obs_by_q["good"])

    rows = []
    share_frames = []
    interval_rows = []
    id_rows = []
    RESULT.mkdir(parents=True, exist_ok=True)

    for name in qualities:
        obs = obs_by_q[name]
        print(f"fitting {name} method={args.method}", flush=True)
        full = fit_asca(world, obs, "full", n_draws=args.draws, seed=args.seed, method=args.method,
                        nuts_warmup=args.nuts_warmup, nuts_samples=args.nuts_samples)
        print(f"  full {full['method']} nll={full['nll']:.1f} {full['note']}", flush=True)
        no_z = fit_asca(world, obs, "no_z", n_draws=0, seed=args.seed, method="map")
        touch = fit_asca(world, obs, "touch", n_draws=0, seed=args.seed, method="map")
        misuse = None
        if name == "good":
            misuse = fit_asca(world, obs, "misuse", n_draws=0, seed=args.seed, method="map")
            print(f"  misuse nll={misuse['nll']:.1f}", flush=True)
        for ch, why in full["prepared"]["reason"].items():
            id_rows.append(dict(quality=name, channel=ch, treatment=why, p_hat=float(obs["p_hat"][world["mkt"].index(ch)])))
        fits = {"asca_full": full, "asca_no_z": no_z, "asca_touch": touch}
        if misuse is not None:
            fits["asca_misuse"] = misuse
        for outcome in ("pipeline", "revenue"):
            truth_v = truth["vector_pipeline"] if outcome == "pipeline" else truth["vector_revenue"]
            packed = baseline_shares(world, obs, outcome)
            positive = obs["y_obs"] if outcome == "pipeline" else obs["w_obs"]
            for bname, blob in packed.items():
                lo, hi = _boot(blob["credit"], world["acv"], positive, args.boot, args.seed + 3)
                rows.append(_row(name, bname, outcome, blob["share"], truth_v, lo, hi,
                                 extra={"unattributed_dollar_fraction": blob["unattributed_dollar_fraction"],
                                        "interval": "bootstrap"}))
                for c, (p, t, a, b) in enumerate(zip(blob["share"], truth_v, lo, hi)):
                    share_frames.append(dict(quality=name, outcome=outcome, model=bname,
                                             channel=SCORE_CHANNELS[c], share=float(p), truth=float(t),
                                             lo=float(a), hi=float(b)))
            ident_mask = [full["prepared"]["identified"][c] for c in SCORE_CHANNELS]
            for model_name, fit in fits.items():
                pred = _point_share(fit, world["acv"], outcome)
                lo = hi = None
                if fit["draws"] is not None and fit["method"] in ("nuts", "laplace") and fit["draws"].shape[0] > 1:
                    samples = draw_shares(fit, world["acv"], outcome)
                    lo = np.quantile(samples, 0.05, axis=0)
                    hi = np.quantile(samples, 0.95, axis=0)
                    for c, (a, b) in enumerate(zip(lo, hi)):
                        interval_rows.append(dict(quality=name, outcome=outcome, model=model_name,
                                                  channel=SCORE_CHANNELS[c], lo=float(a), hi=float(b),
                                                  truth=float(truth_v[c]), identified=bool(ident_mask[c])))
                rows.append(_row(name, model_name, outcome, pred, truth_v, lo, hi, ident_mask,
                                 extra={"interval": fit["method"], "fit_note": fit["note"]}))
                for c, p in enumerate(pred):
                    share_frames.append(dict(quality=name, outcome=outcome, model=model_name,
                                             channel=SCORE_CHANNELS[c], share=float(p), truth=float(truth_v[c]),
                                             lo=float(lo[c]) if lo is not None else np.nan,
                                             hi=float(hi[c]) if hi is not None else np.nan))
            print(f"  {outcome} ASCA MAE={rows[-1]['MAE'] if False else _last_mae(rows, name, 'asca_full', outcome):.4f}", flush=True)

    scores = pd.DataFrame(rows)
    shares = pd.DataFrame(share_frames)
    scores.to_csv(RESULT / "scores.csv", index=False)
    shares.to_csv(RESULT / "shares.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(RESULT / "intervals.csv", index=False)
    pd.DataFrame(id_rows).to_csv(RESULT / "identification.csv", index=False)
    _write_summary(world, truth, scores, shares, obs_by_q)
    _plot(scores)
    print(scores.pivot_table(index=["outcome", "model"], columns="quality", values="MAE").round(4).to_string(), flush=True)
    print("wrote", RESULT, flush=True)


def _point_share(fit, acv, outcome):
    from b2b.model import shares_for_coef
    return shares_for_coef(fit["prepared"], fit["coef"], acv, outcome)


def _last_mae(rows, quality, model, outcome):
    for row in reversed(rows):
        if row["quality"] == quality and row["model"] == model and row["outcome"] == outcome:
            return row["MAE"]
    return float("nan")


def _write_summary(world, truth, scores, shares, obs_by_q):
    summary = {
        "n": int(world["n"]),
        "seed": int(world["seed"]),
        "opportunity_rate": float(world["y"].mean()),
        "wins": int(world["w"].sum()),
        "opportunities": int(world["y"].sum()),
        "win_rate_given_opportunity": float(world["w"][world["y"]].mean()) if world["y"].any() else None,
        "agent_share_of_opportunities": float(world["agent"][world["y"]].mean()) if world["y"].any() else None,
        "ai_executed_share_of_opportunities": float(world["ai_executed"][world["y"]].mean()) if world["y"].any() else None,
        "iab_counts": {k: int((world["iab"] == k).sum()) for k in ("human_owned", "ai_assisted", "ai_executed")},
        "mean_contacts": float(world["n_contacts"].mean()),
        "mean_acv": float(world["acv"].mean()),
        "incremental_pipeline_dollars": truth["incremental_pipeline_dollars"],
        "incremental_revenue_dollars": truth["incremental_revenue_dollars"],
        "pipeline_truth": truth["pipeline"],
        "revenue_truth": truth["revenue"],
        "sales_pipeline_share": truth["sales_pipeline_share"],
        "visibility": {q: {c: float(p) for c, p in zip(
            ["paid_search", "linkedin", "content", "events", "sdr", "partner", "review", "organic", "plg", "abm",
             "llm_research", "wom", "podcast", "sales_ae"], visibility(QUALITIES[q]))}
            for q in QUALITIES},
    }
    (RESULT / "summary.json").write_text(json.dumps(summary, indent=2))
    # Markdown tables at 4 decimals, taken from the frames so the README can copy them.
    lines = ["# B2B run tables", ""]
    lines.append(f"N={summary['n']}, opportunities={summary['opportunities']} ({summary['opportunity_rate']:.4f}), "
                 f"wins={summary['wins']} (win|opp={summary['win_rate_given_opportunity']:.4f}), "
                 f"agent among opportunities={summary['agent_share_of_opportunities']:.4f}, "
                 f"AI-executed demo among opportunities={summary['ai_executed_share_of_opportunities']:.4f}.")
    lines.append("")
    show = scores.copy()
    for col in ("MAE", "MaxAE", "Spearman", "Kendall", "coverage_90", "coverage_90_identified"):
        if col in show:
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    for outcome in ("pipeline", "revenue"):
        lines.append(f"## {outcome}")
        lines.append("")
        cols = ["quality", "model", "MAE", "MaxAE", "Spearman", "Kendall",
                "coverage_90", "coverage_90_identified"]
        cols = [c for c in cols if c in show.columns]
        sub = show[show.outcome == outcome][cols]
        lines.append(sub.to_markdown(index=False))
        lines.append("")
    (RESULT / "scores.md").write_text("\n".join(lines))


def _plot(scores):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    order = ["near_perfect", "good", "degraded", "heavy"]
    models = ["asca_full", "asca_no_z", "asca_touch", "w_shaped", "lead_source", "last_touch", "markov", "set_cr"]
    for ax, outcome in zip(axes, ("pipeline", "revenue")):
        sub = scores[scores.outcome == outcome]
        for model in models:
            s = sub[sub.model == model].set_index("quality").reindex(order)
            ax.plot(order, s["MAE"], marker="o", label=BASELINE_LABELS.get(model, model))
        ax.set_title(outcome)
        ax.set_xlabel("tracking quality")
        ax.set_ylabel("MAE vs causal share")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULT / "tracking_sweep.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
