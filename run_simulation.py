"""Agentic simulation with an explicit agent node, dark influence, a confounder and a non-causal
'agent_checkout' final step. Checks whether ASCA recovers the true interventional (Shapley) shares."""
import json, time
import numpy as np, pandas as pd
from asca.common import Paths, score, per_journey_mae, normalize_rows
from asca import baselines as B
from asca.model import ASCA

SEED, N, OUT, N_DRAWS = 11, 200_000, "results", 200
rng = np.random.default_rng(SEED)
CLICK = ["paid_search", "social", "display", "email", "organic_search", "direct", "referral"]
DARK = ["ai_answer_content"]          # brand content surfaced inside LLM answers: logged impressions, no clicks
EXPO = CLICK + DARK                   # what ASCA sees as exposures X
OBS = CLICK + ["agent_checkout"]      # what path-based models see in the clickstream
ALL = EXPO + ["agent_checkout"]       # scoring vector (truth for agent_checkout = 0)
K = len(EXPO)

# ---------------- true structural parameters ----------------
H, KAPPA = 168.0, 1.0
mix = np.array([0.9, 0.85, 0.8, 0.5, 0.6, 0.4, 0.3])
b_true = np.array([0.9, 0.5, 0.15, 0.6, 0.4, 0.5, 0.2, 0.0])      # direct X->Y (dark channel: 0)
g_true = np.array([0.2, 0.4, 0.0, 0.0, 1.0, 0.0, 0.9, 2.5])        # X->A (agent recommendation propensity)
b0, zY, d, a0, zA, q = -3.6, 1.0, 1.6, -2.2, 0.5, 0.75
truth_params = dict(base_logit_Y=b0, b=dict(zip(EXPO, b_true)), d_agent_on_Y=d, a0=a0, g=dict(zip(EXPO, g_true)),
                    zY=zY, zA=zA, q_agent_checkout=q, half_life=H, kappa=KAPPA)

# ---------------- simulate ----------------
Zc = rng.binomial(1, 0.3, N)                                           # observed high-intent segment (confounder)
n_t = 1 + rng.poisson(1.7, N)
rows, chs, ts = [], [], []
jj = np.repeat(np.arange(N), n_t); rows.append(jj); chs.append(rng.choice(7, len(jj), p=mix / mix.sum())); ts.append(rng.uniform(0, 720, len(jj)))
n_rt = rng.poisson(1.5 * Zc)                                           # retargeting/brand touches driven by Z
jj = np.repeat(np.arange(N), n_rt); rows.append(jj); chs.append(rng.choice([0, 3, 5], len(jj), p=[.3, .4, .3])); ts.append(rng.uniform(0, 720, len(jj)))
n_dark = rng.poisson(0.7, N)
jj = np.repeat(np.arange(N), n_dark); rows.append(jj); chs.append(np.full(len(jj), 7)); ts.append(rng.uniform(0, 720, len(jj)))
J, C, T = map(np.concatenate, (rows, chs, ts))
o = np.lexsort((T, J)); J, C, T = J[o], C[o], T[o]
E = np.zeros((N, K)); np.add.at(E, (J, C), np.exp(-np.log(2) / H * (720 - T))); S = E / (KAPPA + E)
pi = 1 / (1 + np.exp(-(a0 + S @ g_true + zA * Zc)))
A = rng.binomial(1, pi)
p = 1 / (1 + np.exp(-(b0 + S @ b_true + d * A + zY * Zc)))
Y = rng.binomial(1, p)
M = (Y == 1) & (A == 1) & (rng.random(N) < q)                          # observed AI-executed checkout
print(f"N={N} conv_rate={Y.mean():.4f} agent_share_of_conv={A[Y==1].mean():.3f} agent_checkout_share={M[Y==1].mean():.3f}")

# exposure table for ASCA (incl. dark impressions) and clickstream for baselines (no dark, + agent_checkout)
P_exp = Paths(J, C, T, Y, EXPO, N)
clk = C < 7
Jo = np.r_[J[clk], np.flatnonzero(M)]; Co = np.r_[C[clk], np.full(M.sum(), 7)]; To = np.r_[T[clk], np.full(M.sum(), 720.0)]
keep = np.isin(np.arange(N), Jo)                                       # journeys with >=1 observed click/checkout
o = np.lexsort((To, Jo)); Jo, Co, To = Jo[o], Co[o], To[o]
P_obs = Paths(Jo, Co, To, Y, OBS, N)

# ---------------- ground truth: Shapley of v_j(S)=E[Y|do(X_S=x_jS, X_-S=0), Z_j] under true SCM ----------------
true_m = ASCA(K, n_z=1, latent_agent=True, proxy=True, fixed_shape=(H, KAPPA))
th_true = np.r_[b0, b_true, zY, d, a0, g_true, zA, np.log(q / (1 - q))]
Z2 = Zc[:, None].astype(float)
phi_true = true_m.credit(th_true, P_exp, Z2)
phi_true_dir = true_m.credit(th_true, P_exp, Z2, block_agent_path=True)
truth_rows = normalize_rows(phi_true)
truth = dict(zip(EXPO, truth_rows.mean(0))); truth["agent_checkout"] = 0.0
true_med = 1 - phi_true_dir.sum(0) / phi_true.sum(0)
conv = np.flatnonzero(Y == 1)

def to_all(rows, names):                                               # map rows to the 9-dim ALL vector
    out = np.zeros((rows.shape[0], len(ALL)))
    for i, c in enumerate(names): out[:, ALL.index(c)] = rows[:, i]
    return out
gt_all = to_all(truth_rows, EXPO)

# ---------------- baselines on the observable clickstream ----------------
models, rows_by = {}, {}
conv_obs_mask = keep[conv]          # converters with no observed touch get no credit from path models
for name, fn in [("Last-click", B.last_click), ("First-click", B.first_click), ("Linear", B.linear),
                 ("Time-decay (h=168h)", B.time_decay), ("Position-based 40/20/40", B.position_based),
                 ("Shapley-Zhao (counts)", B.shapley_zhao_counts), ("Shapley set-CR", B.shapley_set_cr)]:
    r = to_all(fn(P_obs), OBS); rows_by[name] = r
    models[name] = dict(zip(ALL, r[conv_obs_mask].mean(0)))
mr, magg = B.markov_removal(P_obs)
rows_by["Markov removal"] = to_all(mr, OBS); models["Markov removal"] = dict(zip(ALL, to_all(magg[None], OBS)[0]))

# ---------------- ASCA variants ----------------
variants = [("ASCA v0 (A latent + agent_checkout proxy, Z adjusted)", dict(latent_agent=True, proxy=True), True, True),
            ("ASCA ablation: A latent, no proxy", dict(latent_agent=True, proxy=False), True, False),
            ("ASCA ablation: no A node", dict(latent_agent=False), True, False),
            ("ASCA ablation: no confounder adjustment", dict(latent_agent=True, proxy=True), False, True),
            ("ASCA misuse: agent_checkout as exposure, no A", None, True, False)]
ints, fits, med = {}, {}, {}
for name, kw, useZ, useM in variants:
    t0 = time.time(); print("fitting", name)
    Zf = Z2 if useZ else None
    if kw is None:   # treat agent_checkout as if it were a marketing exposure (post-treatment / collider error)
        P_mis = Paths(np.r_[J, np.flatnonzero(M)], np.r_[C, np.full(M.sum(), 8)], np.r_[T, np.full(M.sum(), 720.)], Y, ALL, N)
        o2 = np.lexsort((P_mis.t, P_mis.jidx)); P_mis = Paths(P_mis.jidx[o2], P_mis.ch[o2], P_mis.t[o2], Y, ALL, N)
        m = ASCA(len(ALL), n_z=1, latent_agent=False).fit(P_mis, Z=Zf, verbose=False)
        phi = m.credit(m.theta, P_mis, Zf); rows = normalize_rows(phi); names = ALL
        draws = m.draws(N_DRAWS, seed=SEED)
        sh = np.array([normalize_rows(m.credit(t, P_mis, Zf)).mean(0) for t in draws])
    else:
        m = ASCA(K, n_z=1 if useZ else 0, **kw).fit(P_exp, Z=Zf, M=M.astype(float) if useM else None, verbose=False)
        phi = m.credit(m.theta, P_exp, Zf); rows = normalize_rows(phi); names = EXPO
        draws = m.draws(N_DRAWS, seed=SEED)
        sh = np.array([normalize_rows(m.credit(t, P_exp, Zf)).mean(0) for t in draws])
        if kw.get("latent_agent"):
            phid = m.credit(m.theta, P_exp, Zf, block_agent_path=True)
            med[name] = dict(zip(EXPO, (1 - phid.sum(0) / phi.sum(0)).tolist()))
    rows_by[name] = to_all(rows, names); models[name] = dict(zip(ALL, rows_by[name].mean(0)))
    sh_all = to_all(sh, names)
    ints[name] = {c: (float(np.quantile(sh_all[:, i], .05)), float(np.quantile(sh_all[:, i], .95))) for i, c in enumerate(ALL)}
    fits[name] = {k: [float(a), float(b)] for k, (a, b) in m.summary().items()}
    print(f"  {time.time()-t0:.0f}s")

tab = []
for name in models:
    s = score(models[name], truth, ALL)
    r = dict(model=name, **s, per_journey_MAE=per_journey_mae(rows_by[name], gt_all),
             credit_to_agent_checkout=models[name]["agent_checkout"], credit_to_dark_channel=models[name]["ai_answer_content"])
    if name in ints:
        r["90%_coverage"] = np.mean([ints[name][c][0] <= truth[c] <= ints[name][c][1] for c in ALL])
    tab.append(r)
tab = pd.DataFrame(tab).sort_values("MAE")
shares = pd.DataFrame(models).T[ALL]; shares.loc["TRUE (interventional Shapley)"] = [truth[c] for c in ALL]
iv = pd.DataFrame({n: {c: f"[{lo:.3f}, {hi:.3f}]" for c, (lo, hi) in d.items()} for n, d in ints.items()})
medtab = pd.DataFrame(med); medtab["TRUE"] = true_med
tab.to_csv(f"{OUT}/simulation_scores.csv", index=False); shares.to_csv(f"{OUT}/simulation_shares.csv")
iv.to_csv(f"{OUT}/simulation_intervals.csv"); medtab.to_csv(f"{OUT}/simulation_mediated_fraction.csv")
with open(f"{OUT}/simulation_scores.md", "w") as f:
    f.write(f"N={N}, conversions={int(Y.sum())} ({Y.mean():.4f}), agent-mediated conversions={A[Y==1].mean():.3f}, "
            f"observed agent_checkout among conversions={M[Y==1].mean():.3f}\n\n" + tab.to_markdown(index=False, floatfmt=".4f")
            + "\n\n### Channel shares\n\n" + shares.to_markdown(floatfmt=".4f") + "\n\n### ASCA 90% Laplace-posterior intervals\n\n"
            + iv.to_markdown() + "\n\n### Fraction of each channel's credit flowing through the agent node A\n\n" + medtab.to_markdown(floatfmt=".3f") + "\n")
json.dump(dict(truth_params=truth_params, fits=fits, n=N, conv=int(Y.sum()), agent_share=float(A[Y == 1].mean()),
               checkout_share=float(M[Y == 1].mean())), open(f"{OUT}/simulation_fit.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(tab.to_string()); print(shares.round(3).to_string()); print(iv.to_string()); print(medtab.round(3).to_string())
