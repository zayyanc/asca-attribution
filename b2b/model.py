"""ASCA-B2B.

Observed exposure is a thinned, sometimes mis-assigned measurement of true exposure.
Keep rates are estimated on a gold panel (production log vs true exposure on the same
accounts) and used to inflate the touch log. LinkedIn and ABM enter as noisy
account-level impression totals. LLM research has almost no referrer; a randomized
account holdout is the exposure (0 on the holdout, the panel's mean dose otherwise).
Channels whose keep rate is below the threshold, and which have neither an aggregate
nor a holdout, are left at zero exposure and are reported as unidentified.

The outcome model is the same two-equation SCM as the simulation: opportunity, then
win given opportunity, with a latent agent marginalized out. The AI-executed demo
flag is a proxy for the agent, not a channel. Sales calls enter only the win
equation, and only calls logged after the decision date. Firmographics are the
backdoor set. Published credit is interventional Shapley on the reconstructed
exposures. NUTS (NumPyro) supplies posterior draws when it samples cleanly;
otherwise draws come from a Laplace approximation at the MAP.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from b2b.common import (
    AGENT_IDX,
    IDX,
    K_MKT,
    MKT,
    SALES,
    TOUCH_ID_THRESHOLD,
    mix_prob,
    outcome_values,
    saturate,
    shares_from_phi,
)

N_G = 4  # agent channel loadings


def _layout(nz: int) -> dict:
    # b0, b[K], zY[nz], d_opp, a0, g[4], zA[nz], logit_q, c0, c[K], zW[nz], d_win, delta
    i = 0
    spec = {}
    def take(name, k):
        nonlocal i
        spec[name] = (i, i + k)
        i += k
    take("b0_opp", 1)
    take("b_opp", K_MKT)
    take("z_opp", nz)
    take("d_opp", 1)
    take("a0", 1)
    take("g", N_G)
    take("z_agent", nz)
    take("logit_q", 1)
    take("b0_win", 1)
    take("b_win", K_MKT)
    take("z_win", nz)
    take("d_win", 1)
    take("delta_sales", 1)
    spec["n"] = i
    return spec


def _unpack(theta, spec):
    def sl(name):
        a, b = spec[name]
        return theta[a:b]
    return dict(
        b0_opp=float(sl("b0_opp")[0]),
        b_opp=np.asarray(sl("b_opp"), dtype=float),
        z_opp=np.asarray(sl("z_opp"), dtype=float),
        d_opp=float(sl("d_opp")[0]),
        a0=float(sl("a0")[0]),
        g=np.asarray(sl("g"), dtype=float),
        z_agent=np.asarray(sl("z_agent"), dtype=float),
        logit_q=float(sl("logit_q")[0]),
        b0_win=float(sl("b0_win")[0]),
        b_win=np.asarray(sl("b_win"), dtype=float),
        z_win=np.asarray(sl("z_win"), dtype=float),
        d_win=float(sl("d_win")[0]),
        delta_sales=float(sl("delta_sales")[0]),
    )


def _prior(theta, spec):
    pen = 0.0
    intercepts = ("b0_opp", "a0", "b0_win")
    for name, span in spec.items():
        if name == "n":
            continue
        a, b = span
        sd = 5.0 if name in intercepts else (1.5 if name == "logit_q" else 2.0)
        pen += 0.5 * np.sum((theta[a:b] / sd) ** 2)
    return pen


def _nll(theta, spec, s, z, s_sales_obs, y, w, proxy, misuse_beta_slot=None):
    c = _unpack(theta, spec)
    g_full = np.zeros(K_MKT)
    g_full[AGENT_IDX] = c["g"]
    lp = c["a0"] + s @ g_full + z @ c["z_agent"]
    eta = c["b0_opp"] + s @ c["b_opp"] + z @ c["z_opp"]
    if misuse_beta_slot is not None:
        # misuse_beta_slot is the already-scaled offset (flag * coefficient), passed separately
        # so it is not inside theta twice. The runner adds it only for the misuse fit via z.
        pass
    l_pi1 = log_expit(lp)
    l_pi0 = log_expit(-lp)
    lq = c["logit_q"]
    # Opportunity, with the AI-executed flag as a proxy: M=1 only if A=1.
    l1m = l_pi1 + log_expit(eta + c["d_opp"]) + log_expit(lq)
    l1n = np.logaddexp(l_pi0 + log_expit(eta), l_pi1 + log_expit(eta + c["d_opp"]) + log_expit(-lq))
    l0 = np.logaddexp(l_pi0 + log_expit(-eta), l_pi1 + log_expit(-(eta + c["d_opp"])))
    ll_opp = np.where(y == 0, l0, np.where(proxy == 1, l1m, l1n)).sum()
    # Win equation on observed opportunities only.
    won_rows = y == 1
    ll_win = 0.0
    if won_rows.any():
        eta_w = (c["b0_win"] + s[won_rows] @ c["b_win"] + z[won_rows] @ c["z_win"]
                 + c["delta_sales"] * s_sales_obs[won_rows])
        lp_w = lp[won_rows]
        p_w = mix_prob(eta_w, c["d_win"], lp_w)
        ww = w[won_rows].astype(float)
        p_w = np.clip(p_w, 1e-8, 1 - 1e-8)
        ll_win = np.sum(ww * np.log(p_w) + (1 - ww) * np.log(1 - p_w))
    return -(ll_opp + ll_win) + _prior(theta, spec)


def _llm_exposure(world) -> np.ndarray:
    """Dose for the dark LLM channel.

    The touch log is not used. Holdout accounts are set to zero (the experiment).
    Gold-panel accounts use the exposure the panel actually measured. Everyone else
    gets the firmographic prediction from that panel, so the dose is not a single
    constant and is not copied from the hidden true exposure.
    """
    c = IDX["llm_research"]
    z = world["z"]
    hold = world["hold_llm"]
    panel = world["gold"] & ~hold
    y = world["e"][:, c]
    n = world["n"]
    if panel.sum() < z.shape[1] + 2:
        pred = np.full(n, float(y[panel].mean()) if panel.any() else 0.3)
    else:
        x = np.column_stack([np.ones(int(panel.sum())), z[panel]])
        coef, *_ = np.linalg.lstsq(x, y[panel], rcond=None)
        pred = np.column_stack([np.ones(n), z]) @ coef
    pred = np.clip(pred, 0.0, None)
    pred[panel] = y[panel]
    pred[hold] = 0.0
    return pred


def _sales_potential(z, s_obs, y):
    """Predict the sales intensity an account would field if it became an opportunity."""
    rows = y == 1
    if rows.sum() < z.shape[1] + 2:
        return np.clip(s_obs, 0, 1)
    x = np.column_stack([np.ones(rows.sum()), z[rows]])
    coef, *_ = np.linalg.lstsq(x, s_obs[rows], rcond=None)
    pred = np.column_stack([np.ones(z.shape[0]), z]) @ coef
    pred = np.clip(pred, 0.0, 0.95)
    # Where the CRM logged post-decision calls, use that intensity, idiosyncrasy included.
    pred[rows] = np.clip(s_obs[rows], 0.0, 0.95)
    return pred


def _calibrate(world, obs) -> np.ndarray:
    """E[true exposure | production log, aggregates, firmographics], fit on the gold panel."""
    n = world["n"]
    gold = world["gold"]
    z = world["z"]
    e_hat = np.zeros((n, K_MKT))
    if gold.sum() < 20:
        return obs["e_obs"].copy()
    for c, name in enumerate(MKT):
        cols = [np.ones(int(gold.sum())), obs["e_obs"][gold, c]]
        if name in ("linkedin", "abm"):
            cols.append(obs["e_agg"][gold, c])
        cols.append(z[gold])
        coef, *_ = np.linalg.lstsq(np.column_stack(cols), world["e"][gold, c], rcond=None)
        cols_all = [np.ones(n), obs["e_obs"][:, c]]
        if name in ("linkedin", "abm"):
            cols_all.append(obs["e_agg"][:, c])
        cols_all.append(z)
        e_hat[:, c] = np.clip(np.column_stack(cols_all) @ coef, 0, None)
    e_hat[world["hold_abm"], IDX["abm"]] = 0.0
    e_hat[world["hold_llm"], IDX["llm_research"]] = 0.0
    return e_hat


def design(world, obs, mode: str):
    """Reconstruct saturated exposures. `mode` is full, no_z, touch, or misuse."""
    n = world["n"]
    reason = {}
    if mode == "touch":
        e_hat = obs["e_obs"].copy()
        for c, name in enumerate(MKT):
            reason[name] = "touch log only" if e_hat[:, c].std() > 1e-8 else "no observed touches"
    else:
        e_hat = _calibrate(world, obs)
        for c, name in enumerate(MKT):
            if name == "llm_research":
                reason[name] = "gold-panel calibration plus the account holdout (touch log is not used)"
            elif name in ("linkedin", "abm"):
                reason[name] = "account-level impressions, calibrated on the gold panel"
            elif obs["p_hat"][c] >= TOUCH_ID_THRESHOLD:
                reason[name] = "touch log, regression-calibrated on the gold panel"
            else:
                e_hat[:, c] = 0.0
                reason[name] = "unidentified without an experiment"
    s = saturate(np.clip(e_hat, 0, None))
    s_obs_sales = obs.get("sales_effort", obs["sales_post_count"] / (1.0 + obs["sales_post_count"]))
    z = world["z"] if mode != "no_z" else np.zeros_like(world["z"])
    s_pot = _sales_potential(z if mode != "no_z" else world["z"], s_obs_sales, obs["y_obs"])
    identified = {name: reason[name] != "unidentified without an experiment" and reason[name] != "no observed touches"
                  for name in MKT}
    if mode == "touch":
        identified = {name: reason[name] == "touch log only" for name in MKT}
    identified[SALES] = True  # post-decision calls are in the CRM
    identified["ai_sourced"] = False
    return dict(s=s, s_sales_obs=s_obs_sales, s_sales_pot=s_pot, z=z, reason=reason, identified=identified)


def _init_theta(spec, y):
    theta = np.zeros(spec["n"])
    rate = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    a, b = spec["b0_opp"]
    theta[a] = np.log(rate / (1 - rate))
    a, b = spec["a0"]
    theta[a] = -1.2
    a, b = spec["d_opp"]
    theta[a] = 0.4
    a, b = spec["d_win"]
    theta[a] = 0.2
    a, b = spec["b0_win"]
    theta[a] = -0.2
    return theta


def fit_map(prepared, obs, misuse_flag: bool = False):
    spec = _layout(prepared["z"].shape[1])
    y = obs["y_obs"].astype(float)
    w = obs["w_obs"].astype(float)
    proxy = obs["ai_flag"].astype(float)
    s = prepared["s"]
    z = prepared["z"]
    # Misuse appends the AI flag as an extra firmographic-style column on the opportunity equation.
    if misuse_flag:
        z = np.column_stack([z, obs["ai_flag"].astype(float)])
        spec = _layout(z.shape[1])
    s_obs = prepared["s_sales_obs"]
    theta0 = _init_theta(spec, y)

    def fun(th):
        return _nll(th, spec, s, z, s_obs, y, w, proxy)

    bounds = []
    for name, span in spec.items():
        if name == "n":
            continue
        a, b = span
        free = name not in ("b_opp", "b_win", "g", "d_opp", "d_win", "delta_sales")
        bounds.extend([(None, None) if free else (0.0, None)] * (b - a))
    opt = minimize(fun, theta0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=400, ftol=1e-8))
    return spec, opt, z


def _laplace_draws(fun, theta, n_draws, seed, rel=1e-4):
    k = theta.size
    h = np.zeros((k, k))
    f0 = fun(theta)
    step = rel * np.maximum(1.0, np.abs(theta))
    for i in range(k):
        for j in range(i, k):
            if i == j:
                tp = theta.copy(); tp[i] += step[i]
                tm = theta.copy(); tm[i] -= step[i]
                h[i, i] = (fun(tp) - 2 * f0 + fun(tm)) / step[i] ** 2
            else:
                v = []
                for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    t = theta.copy()
                    t[i] += si * step[i]
                    t[j] += sj * step[j]
                    v.append(fun(t))
                h[i, j] = h[j, i] = (v[0] - v[1] - v[2] + v[3]) / (4 * step[i] * step[j])
    h = 0.5 * (h + h.T)
    # Likelihood curvature can be tiny on unidentified coefficients. Keep the prior's curvature.
    try:
        cov = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(h)
    cov = 0.5 * (cov + cov.T)
    eig, vec = np.linalg.eigh(cov)
    cov = (vec * np.clip(eig, 1e-8, 1e3)) @ vec.T
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(theta, cov, size=n_draws), cov


def _nuts_draws(spec, theta, prepared, obs, z, n_warmup, n_samples, seed):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_value

    numpyro.set_host_device_count(1)
    s = jnp.asarray(prepared["s"])
    zz = jnp.asarray(z)
    y = jnp.asarray(obs["y_obs"].astype(float))
    w = jnp.asarray(obs["w_obs"].astype(float))
    proxy = jnp.asarray(obs["ai_flag"].astype(float))
    s_obs = jnp.asarray(prepared["s_sales_obs"])
    agent_idx = jnp.asarray(AGENT_IDX)
    layout = {name: spec[name] for name in spec if name != "n"}

    def model():
        parts = []
        ordered = []
        for name, (a, b) in layout.items():
            sd = 5.0 if name in ("b0_opp", "a0", "b0_win") else (1.5 if name == "logit_q" else 2.0)
            draw = numpyro.sample(name, dist.Normal(0, sd).expand([b - a]).to_event(1))
            parts.append(draw)
            ordered.append(name)
        theta_j = jnp.concatenate(parts)
        # Rebuild in layout order, which matches insertion order of _layout.
        c = {}
        cursor = 0
        for name, (a, b) in layout.items():
            c[name] = theta_j[cursor:cursor + (b - a)]
            cursor += b - a
        g = jnp.zeros(K_MKT).at[agent_idx].set(c["g"][:, 0] if False else c["g"])
        lp = c["a0"][0] + s @ g + zz @ c["z_agent"]
        eta = c["b0_opp"][0] + s @ c["b_opp"] + zz @ c["z_opp"]
        l_pi1 = jax.nn.log_sigmoid(lp)
        l_pi0 = jax.nn.log_sigmoid(-lp)
        lq = c["logit_q"][0]
        d = c["d_opp"][0]
        l1m = l_pi1 + jax.nn.log_sigmoid(eta + d) + jax.nn.log_sigmoid(lq)
        l1n = jnp.logaddexp(l_pi0 + jax.nn.log_sigmoid(eta), l_pi1 + jax.nn.log_sigmoid(eta + d) + jax.nn.log_sigmoid(-lq))
        l0 = jnp.logaddexp(l_pi0 + jax.nn.log_sigmoid(-eta), l_pi1 + jax.nn.log_sigmoid(-(eta + d)))
        ll = jnp.where(y == 0, l0, jnp.where(proxy == 1, l1m, l1n))
        numpyro.factor("opp", ll.sum())
        won = y == 1
        eta_w = (c["b0_win"][0] + s @ c["b_win"] + zz @ c["z_win"] + c["delta_sales"][0] * s_obs)
        p_w = mix_prob_jax(eta_w, c["d_win"][0], lp)
        p_w = jnp.clip(p_w, 1e-6, 1 - 1e-6)
        llw = w * jnp.log(p_w) + (1 - w) * jnp.log(1 - p_w)
        numpyro.factor("win", jnp.where(won, llw, 0.0).sum())

    def mix_prob_jax(eta, d, lp):
        pi = jax.nn.sigmoid(lp)
        return (1 - pi) * jax.nn.sigmoid(eta) + pi * jax.nn.sigmoid(eta + d)

    init = {}
    for name, (a, b) in layout.items():
        init[name] = np.asarray(theta[a:b], dtype=np.float32)
    kernel = NUTS(model, init_strategy=init_to_value(values=init), target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    samples = mcmc.get_samples()
    diverging = int(np.asarray(mcmc.get_extra_fields().get("diverging", np.array([False]))).sum()) if False else 0
    # extra fields need collect. Re-run is wasteful; check divergences via get_extra_fields after enabling.
    stacked = []
    for name, (a, b) in layout.items():
        stacked.append(np.asarray(samples[name]))
    draws = np.concatenate(stacked, axis=1)
    return draws, diverging


def fit_asca(world, obs, mode: str = "full", n_draws: int = 40, seed: int = 0, method: str = "laplace",
             nuts_warmup: int = 200, nuts_samples: int = 80):
    prepared = design(world, obs, "full" if mode == "misuse" else mode)
    misuse = mode == "misuse"
    spec, opt, z = fit_map(prepared, obs, misuse_flag=misuse)
    prepared = dict(prepared)
    prepared["z"] = z
    theta = opt.x
    used = method
    note = ""
    draws = None
    if n_draws <= 0 or method == "map":
        draws = theta[None, :]
        used = "map"
        note = "MAP only; no posterior draws"
        method = "map"
    if method == "nuts":
        try:
            # Collect divergences.
            draws, div = _nuts_with_divergences(spec, theta, prepared, obs, z, nuts_warmup, nuts_samples, seed)
            if div > 0.1 * nuts_samples:
                note = f"NUTS had {div} divergences out of {nuts_samples}; using Laplace draws instead"
                draws = None
                used = "laplace"
            else:
                note = f"NUTS warmup {nuts_warmup}, samples {nuts_samples}, divergences {div}"
                used = "nuts"
        except Exception as exc:  # noqa: BLE001 — fall back and record why
            note = f"NUTS failed ({type(exc).__name__}: {exc}); using Laplace draws"
            used = "laplace"
            draws = None
    if draws is None:
        def fun(th):
            return _nll(th, spec, prepared["s"], z, prepared["s_sales_obs"],
                        obs["y_obs"].astype(float), obs["w_obs"].astype(float), obs["ai_flag"].astype(float))
        draws, _cov = _laplace_draws(fun, theta, n_draws, seed)
        if used != "nuts":
            used = "laplace"
    coef_map = _unpack(theta, spec)
    return dict(spec=spec, theta=theta, draws=draws, coef=coef_map, prepared=prepared,
                method=used, note=note, success=bool(opt.success), nll=float(opt.fun), mode=mode)


def _nuts_with_divergences(spec, theta, prepared, obs, z, n_warmup, n_samples, seed):
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.initialization import init_to_value

    s = jnp.asarray(prepared["s"])
    zz = jnp.asarray(z)
    y = jnp.asarray(obs["y_obs"].astype(np.float32))
    w = jnp.asarray(obs["w_obs"].astype(np.float32))
    proxy = jnp.asarray(obs["ai_flag"].astype(np.float32))
    s_obs = jnp.asarray(prepared["s_sales_obs"])
    agent_idx = jnp.asarray(AGENT_IDX)
    layout = [(name, spec[name]) for name in spec if name != "n"]

    def model():
        pieces = []
        for name, (a, b) in layout:
            sd = 5.0 if name in ("b0_opp", "a0", "b0_win") else (1.5 if name == "logit_q" else 2.0)
            pieces.append(numpyro.sample(name, dist.Normal(0, sd).expand([b - a]).to_event(1)))
        c = {name: pieces[i] for i, (name, _) in enumerate(layout)}
        g = jnp.zeros((K_MKT,)).at[agent_idx].set(c["g"])
        lp = c["a0"][0] + s @ g + zz @ c["z_agent"]
        eta = c["b0_opp"][0] + s @ c["b_opp"] + zz @ c["z_opp"]
        l_pi1 = jax.nn.log_sigmoid(lp)
        l_pi0 = jax.nn.log_sigmoid(-lp)
        lq = c["logit_q"][0]
        d = c["d_opp"][0]
        l1m = l_pi1 + jax.nn.log_sigmoid(eta + d) + jax.nn.log_sigmoid(lq)
        l1n = jnp.logaddexp(l_pi0 + jax.nn.log_sigmoid(eta), l_pi1 + jax.nn.log_sigmoid(eta + d) + jax.nn.log_sigmoid(-lq))
        l0 = jnp.logaddexp(l_pi0 + jax.nn.log_sigmoid(-eta), l_pi1 + jax.nn.log_sigmoid(-(eta + d)))
        ll = jnp.where(y == 0, l0, jnp.where(proxy == 1, l1m, l1n))
        numpyro.factor("opp", jnp.sum(ll))
        eta_w = c["b0_win"][0] + s @ c["b_win"] + zz @ c["z_win"] + c["delta_sales"][0] * s_obs
        pi = jax.nn.sigmoid(lp)
        p_w = (1 - pi) * jax.nn.sigmoid(eta_w) + pi * jax.nn.sigmoid(eta_w + c["d_win"][0])
        p_w = jnp.clip(p_w, 1e-6, 1 - 1e-6)
        llw = w * jnp.log(p_w) + (1.0 - w) * jnp.log(1.0 - p_w)
        numpyro.factor("win", jnp.sum(jnp.where(y == 1, llw, 0.0)))

    init = {name: np.asarray(theta[a:b]) for name, (a, b) in layout}
    kernel = NUTS(model, init_strategy=init_to_value(values=init), target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=1,
                progress_bar=False, jit_model_args=True)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    div = int(np.asarray(extra["diverging"]).sum()) if "diverging" in extra else 0
    draws = np.concatenate([np.asarray(samples[name]) for name, _ in layout], axis=1)
    return draws, div


def shares_for_coef(prepared, coef, acv, outcome: str) -> np.ndarray:
    phi = outcome_values(prepared["s"], prepared["s_sales_pot"], prepared["z"], acv, coef, outcome)
    share = shares_from_phi(phi)
    # ai_sourced is not given a structural exposure in the honest model.
    return np.r_[share, 0.0]


def draw_shares(fit, acv, outcome: str, max_draws: int | None = None) -> np.ndarray:
    draws = fit["draws"]
    if max_draws is not None:
        draws = draws[:max_draws]
    spec = fit["spec"]
    prepared = fit["prepared"]
    out = np.zeros((draws.shape[0], len(shares_for_coef(prepared, fit["coef"], acv, outcome))))
    for i, th in enumerate(draws):
        out[i] = shares_for_coef(prepared, _unpack(th, spec), acv, outcome)
    return out
