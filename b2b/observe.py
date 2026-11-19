"""Production tracking layer. Truth stays in the world; models only see this.

Observation depends on tracking quality q in [0, 1]:
- channel-specific keep rates (cookies, CRM, offline, dark)
- identity: matched to the right account, matched to the wrong account, or dropped
- a noisy CRM copy of opportunity, win, and MQA date
- a biased self-reported lead source
- account-level impression totals for LinkedIn and ABM (not touch-level)
- the AI-executed demo flag, which is a descendant of the agent, not an exposure
"""
from __future__ import annotations

import numpy as np

from b2b.common import (
    AGGREGATE_CHANNELS,
    CHANNELS,
    DECISION_DAY,
    IDX,
    K_MKT,
    TOUCH_ID_THRESHOLD,
    visibility,
)


# Self-report bias. Memorable and recent-feeling channels, plus a ChatGPT label, are over-chosen.
_REPORT_BIAS = {
    "paid_search": 0.15,
    "linkedin": 0.25,
    "content": 0.35,
    "events": 1.45,
    "sdr": 1.15,
    "partner": 0.85,
    "review": 0.95,
    "organic": 0.20,
    "plg": 0.15,
    "abm": -0.55,
    "llm_research": -1.60,
    "wom": -0.90,
    "podcast": 1.05,
    "sales_ae": 0.55,
    "chatgpt": 0.35,
}
_REPORT_CATS = list(_REPORT_BIAS)


def observe(world: dict, q: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = world["n"]
    p_keep = visibility(q)
    p_match = 0.50 + 0.45 * q
    p_wrong = 0.06 * (1.0 - q)

    e = world["e"]
    # Aggregate impressions: a noisy account total, still reported when the touch log is empty.
    agg_sigma = 0.08 + 0.45 * (1.0 - q)
    e_agg = np.zeros((n, K_MKT))
    for name in AGGREGATE_CHANNELS:
        c = IDX[name]
        noise = np.exp(rng.normal(0.0, agg_sigma, n))
        e_agg[:, c] = e[:, c] * noise

    touches = world["touches"]
    acc = touches["account_id"]
    ch = touches["channel_idx"]
    keep = rng.random(acc.size) < p_keep[ch]
    fate = rng.random(acc.size)
    matched = keep & (fate < p_match)
    wrong = keep & (fate >= p_match) & (fate < p_match + p_wrong)
    obs_account = acc.copy()
    if wrong.any():
        obs_account[wrong] = rng.integers(0, n, int(wrong.sum()))
    seen = matched | wrong

    # Rebuild observed exposure from surviving touches. Marketing uses role-weighted decay.
    # Sales uses a post-decision count. Wrong-account touches land on the wrong account.
    from b2b.simulate import _decay

    e_obs = np.zeros((n, K_MKT))
    sales_post_count = np.zeros(n)
    sales_any_count = np.zeros(n)
    # CRM effort score: a noisy reading of structural sales intensity on opportunities
    # the CRM managed to record. Path rules do not see this; they count calls.
    sales_effort = np.zeros(n)
    if seen.any():
        oa = obs_account[seen]
        oc = ch[seen]
        od = touches["day"][seen]
        ow = touches["role_weight"][seen]
        opost = touches["post_decision"][seen]
        mkt = oc < K_MKT
        if mkt.any():
            np.add.at(e_obs, (oa[mkt], oc[mkt]), ow[mkt] * _decay(od[mkt]))
        if (~mkt).any():
            np.add.at(sales_any_count, oa[~mkt], 1.0)
            post = (~mkt) & opost
            if post.any():
                np.add.at(sales_post_count, oa[post], 1.0)

    # Gold panel: we observe production AND the true exposure, and use the pairs only to
    # estimate keep rates. True exposures are not copied into the non-panel regression.
    gold = world["gold"]
    p_hat = np.ones(K_MKT)
    if gold.any() and e[gold].sum() > 0:
        p_hat = (e_obs[gold].sum(0) + 2.0) / (e[gold].sum(0) + 2.0)
        p_hat = np.clip(p_hat, 1e-3, 1.0)

    # CRM labels, missing more often as tracking degrades.
    p_miss_opp = 0.02 + 0.08 * (1.0 - q)
    p_miss_win = 0.03 + 0.12 * (1.0 - q)
    y_obs = world["y"].copy()
    w_obs = world["w"].copy()
    y_obs[world["y"] & (rng.random(n) < p_miss_opp)] = False
    w_obs[world["w"] & (rng.random(n) < p_miss_win)] = False
    w_obs[~y_obs] = False
    logged_opp = y_obs
    if logged_opp.any():
        sales_effort[logged_opp] = np.clip(
            world["s_sales"][logged_opp] + rng.normal(0.0, 0.06, int(logged_opp.sum())), 0.0, 0.98)

    p_miss_mqa = 0.15 + 0.45 * (1.0 - q)
    mqa_crm = world["mqa_day"].copy()
    known = np.isfinite(mqa_crm)
    mqa_crm[known & (rng.random(n) < p_miss_mqa)] = np.nan
    # Surviving MQA dates are late by a few days.
    late = np.isfinite(mqa_crm)
    mqa_crm[late] = np.clip(mqa_crm[late] + rng.uniform(0, 14, int(late.sum())), 0, DECISION_DAY)

    first_obs = np.array(["unknown"] * n, dtype=object)
    if seen.any():
        oa = obs_account[seen]
        od = touches["day"][seen]
        oc = ch[seen]
        order = np.lexsort((od, oa))
        oa, oc = oa[order], oc[order]
        is_first = np.r_[True, oa[1:] != oa[:-1]]
        first_obs[oa[is_first]] = np.array(CHANNELS, dtype=object)[oc[is_first]]
    lead = _lead_source(world, q, rng, first_obs)

    # AI-executed flag is logged in the product, so it survives cookie loss. It is not an exposure.
    ai_flag = world["ai_executed"].copy()

    identifiable = p_hat >= TOUCH_ID_THRESHOLD
    for name in AGGREGATE_CHANNELS:
        identifiable[IDX[name]] = True  # account-level impressions still exist
    for name in ("llm_research",):
        # Touch log is not used even if a handful of referrals survive. The holdout is the signal.
        identifiable[IDX[name]] = False

    return dict(
        q=q,
        p_keep=p_keep,
        p_hat=p_hat,
        touch_identifiable=identifiable,
        e_obs=e_obs,
        e_agg=e_agg,
        sales_post_count=sales_post_count,
        sales_any_count=sales_any_count,
        sales_effort=sales_effort,
        y_obs=y_obs,
        w_obs=w_obs,
        mqa_crm=mqa_crm,
        lead_source=lead,
        ai_flag=ai_flag,
        seen_idx=np.flatnonzero(seen),
        obs_account=obs_account,
        touches=touches,
        seen=seen,
    )


def _lead_source(world, q, rng, first_obs: np.ndarray) -> np.ndarray:
    """CRM lead source. Worse tracking means the field falls back to a biased self-report."""
    n = world["n"]
    cats = _REPORT_CATS
    util = np.zeros((n, len(cats)))
    e = world["e"]
    for j, name in enumerate(cats):
        util[:, j] = _REPORT_BIAS[name]
        if name in IDX and IDX[name] < K_MKT:
            util[:, j] += 0.85 * np.log1p(e[:, IDX[name]])
        elif name == "chatgpt":
            util[:, j] += np.where(world["agent"], 1.35, 0.15)
    util -= util.max(axis=1, keepdims=True)
    prob = np.exp(util)
    prob /= prob.sum(axis=1, keepdims=True)
    cdf = np.cumsum(prob, axis=1)
    pick = np.clip((rng.random(n)[:, None] > cdf).sum(axis=1), 0, len(cats) - 1)
    self_report = np.array(cats, dtype=object)[pick]
    self_report = np.where(self_report == "chatgpt", "ai_sourced", self_report)
    use_self = rng.random(n) < (0.25 + 0.55 * (1.0 - q))
    # Accounts with no surviving touch have nothing but the self-report or "unknown".
    logged = np.where(first_obs != "unknown", first_obs, self_report)
    lead = np.where(use_self, self_report, logged)
    return lead.astype(object)
