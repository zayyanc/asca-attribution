"""Seeded B2B structural causal model.

Unit of analysis is the account. Contacts and roles only change how a touch is
weighted when it rolls up to the account. The decision date is day 110. Marketing
touches land before that date. Account-executive sales effort is a post-decision
intensity that depends on firmographics and an idiosyncratic shock, and it enters
only the win equation. A binary latent agent (buyer using an LLM to build the
shortlist) is a mediator for content, review sites, podcasts, and dark LLM research.

Pipeline value of a coalition is ACV * P(opportunity | do(exposure)).
Revenue value is ACV * P(opportunity | do) * P(closed-won | opportunity, do).
Ground-truth credit is the exact Shapley value of those functions. See truth.py.
"""
from __future__ import annotations

import numpy as np

from b2b.common import (
    AGENT_IDX,
    CHANNELS,
    DECISION_DAY,
    HALF_LIFE,
    HORIZON_DAY,
    IDX,
    K_MKT,
    LN2,
    MKT,
    ROLE_WEIGHT,
    ROLES,
    Z_NAMES,
    saturate,
)

# Opportunity and win coefficients on saturated, role-weighted exposure.
# Sales is not in this list; its win coefficient is DELTA_SALES.
B_OPP = np.array([
    0.70,  # paid_search
    0.42,  # linkedin
    0.90,  # content
    0.38,  # events
    0.58,  # sdr
    0.48,  # partner
    0.68,  # review
    0.32,  # organic
    0.95,  # plg
    0.36,  # abm
    1.10,  # llm_research
    0.32,  # wom
    0.28,  # podcast
])
B_WIN = np.array([
    0.12,  # paid_search
    0.10,  # linkedin
    0.28,  # content
    0.58,  # events
    0.06,  # sdr
    0.52,  # partner
    0.36,  # review
    0.08,  # organic
    0.42,  # plg
    0.10,  # abm
    0.48,  # llm_research
    0.18,  # wom
    0.14,  # podcast
])
# Agent loadings on (content, review, llm_research, podcast).
G_AGENT = np.array([0.95, 1.25, 1.55, 0.40])
B0_OPP = -3.85
B0_WIN = -2.15
D_OPP = 0.90
D_WIN = 0.35
A0 = -1.35
DELTA_SALES = 1.45
# Firmographic loadings: icp, intent, existing, mid_market, enterprise.
Z_OPP = np.array([0.90, 0.80, 0.40, 0.05, -0.10])
Z_WIN = np.array([0.55, 0.20, 0.50, 0.10, 0.25])
Z_AGENT = np.array([0.15, 0.45, 0.10, 0.00, 0.05])

# Mean touch counts before targeting and holdouts.
BASE_RATE = np.array([
    1.25, 0.85, 1.55, 0.30, 0.40,
    0.32, 0.50, 0.95, 0.48, 0.50,
    0.75, 0.38, 0.32,
])
# Extra rate multiplier pieces on (icp, intent, existing, enterprise).
# SDR, ABM, and LinkedIn are aimed at high-fit accounts, which also win more often.
TARGET = np.zeros((K_MKT, 4))
TARGET[IDX["sdr"], :] = [1.8, 1.3, 0.9, 0.2]
TARGET[IDX["abm"], :] = [1.7, 0.9, 0.5, 1.1]
TARGET[IDX["linkedin"], :] = [0.9, 0.8, 0.2, 0.5]
TARGET[IDX["events"], :] = [0.15, 0.1, 0.2, 0.9]
TARGET[IDX["content"], :] = [0.1, 0.35, 0.0, 0.1]
TARGET[IDX["review"], :] = [0.05, 0.25, 0.0, 0.0]
TARGET[IDX["llm_research"], :] = [0.1, 0.45, 0.0, 0.15]


def true_coef() -> dict:
    return dict(
        b_opp=B_OPP.copy(), b_win=B_WIN.copy(), g=G_AGENT.copy(),
        b0_opp=B0_OPP, b0_win=B0_WIN, d_opp=D_OPP, d_win=D_WIN, a0=A0,
        delta_sales=DELTA_SALES, z_opp=Z_OPP.copy(), z_win=Z_WIN.copy(), z_agent=Z_AGENT.copy(),
    )


def _decay(days: np.ndarray) -> np.ndarray:
    return np.exp(-LN2 / HALF_LIFE * (DECISION_DAY - days))


def simulate_world(n: int = 5000, seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)
    segment = rng.choice(3, n, p=[0.50, 0.35, 0.15])
    icp = np.clip(rng.beta(2.1, 1.8, n) * 0.75 + 0.25 * (segment / 2.0), 0, 1)
    intent = np.clip(0.65 * rng.beta(1.7, 2.2, n) + 0.35 * icp, 0, 1)
    existing = rng.random(n) < (0.08 + 0.32 * icp + 0.08 * (segment == 2))
    z = np.column_stack([
        icp,
        intent,
        existing.astype(float),
        (segment == 1).astype(float),
        (segment == 2).astype(float),
    ])
    log_acv = np.array([8.35, 9.65, 11.05])[segment] + 0.20 * icp + rng.normal(0, 0.18, n)
    acv = np.exp(log_acv)

    u_draw = rng.random(n)
    hold_abm = u_draw < 0.08
    hold_llm = (u_draw >= 0.08) & (u_draw < 0.16)
    gold = rng.random(n) < 0.10

    n_contacts = np.clip(1 + rng.poisson(1.05 + 0.55 * segment), 1, 6).astype(int)
    feat = np.column_stack([icp, intent, existing.astype(float), (segment == 2).astype(float)])
    rates = BASE_RATE[None, :] * (1.0 + feat @ TARGET.T)
    rates[:, IDX["abm"]] *= ~hold_abm
    rates[:, IDX["llm_research"]] *= ~hold_llm

    e = np.zeros((n, K_MKT))
    touch_account, touch_contact, touch_channel, touch_day, touch_weight = [], [], [], [], []
    contact_rows = []
    for a in range(n):
        for slot in range(int(n_contacts[a])):
            role = ROLES[min(slot, len(ROLES) - 1)]
            contact_rows.append((a, a * 10 + slot, role, ROLE_WEIGHT[role]))

    for c in range(K_MKT):
        counts = rng.poisson(rates[:, c])
        if counts.sum() == 0:
            continue
        acc = np.repeat(np.arange(n), counts)
        days = rng.uniform(0.0, DECISION_DAY, acc.size)
        slots = rng.integers(0, n_contacts[acc])
        roles = np.array([ROLES[min(int(s), len(ROLES) - 1)] for s in slots])
        weights = np.array([ROLE_WEIGHT[r] for r in roles])
        np.add.at(e, (acc, np.full(acc.size, c, dtype=int)), weights * _decay(days))
        touch_account.append(acc)
        touch_contact.append(acc * 10 + slots)
        touch_channel.append(np.full(acc.size, c, dtype=int))
        touch_day.append(days)
        touch_weight.append(weights)

    s = saturate(e)
    g_full = np.zeros(K_MKT)
    g_full[AGENT_IDX] = G_AGENT
    logit_pi = A0 + s @ g_full + z @ Z_AGENT
    pi = 1.0 / (1.0 + np.exp(-logit_pi))
    agent = rng.random(n) < pi
    eta_opp = B0_OPP + s @ B_OPP + D_OPP * agent + z @ Z_OPP
    p_opp = 1.0 / (1.0 + np.exp(-eta_opp))
    y = rng.random(n) < p_opp

    # Potential AE intensity for every account (used if the account is an opportunity).
    # Intensity has a firmographic piece (more senior attention on high-fit accounts)
    # and a large idiosyncratic piece, so the causal effect is not collinear with Z.
    lam = 0.80 + 1.60 * icp + 0.70 * intent + 0.40 * existing.astype(float)
    idiosyncrasy = rng.random(n)
    s_sales = np.clip(0.08 + 0.40 * icp + 0.12 * intent + 0.62 * idiosyncrasy, 0.02, 0.97)
    eta_win = B0_WIN + s @ B_WIN + D_WIN * agent + DELTA_SALES * s_sales + z @ Z_WIN
    p_win = 1.0 / (1.0 + np.exp(-eta_win))
    w = (y == 1) & (rng.random(n) < p_win)

    # Logged sales activity. Post-decision calls track the structural intensity on opportunities.
    # Cold calls before the decision are in the log and are not in the win equation.
    sales_account, sales_day, sales_post = [], [], []
    n_cold = rng.poisson(0.25, n)
    if n_cold.sum():
        acc = np.repeat(np.arange(n), n_cold)
        sales_account.append(acc)
        sales_day.append(rng.uniform(0.0, DECISION_DAY, acc.size))
        sales_post.append(np.zeros(acc.size, dtype=bool))
    n_ae = np.where(y == 1, rng.poisson(np.clip(lam * (0.45 + idiosyncrasy), 0.2, None)), 0)
    if n_ae.sum():
        acc = np.repeat(np.arange(n), n_ae)
        sales_account.append(acc)
        sales_day.append(rng.uniform(DECISION_DAY, HORIZON_DAY, acc.size))
        sales_post.append(np.ones(acc.size, dtype=bool))

    # AI-executed demo request: a descendant of the agent and the opportunity, not a cause.
    q_exec = 0.42
    ai_executed = (y == 1) & (agent == 1) & (rng.random(n) < q_exec)
    iab = np.where(agent == 0, "human_owned", np.where(ai_executed, "ai_executed", "ai_assisted"))

    # Operational stage dates. These are timestamps for the funnel, not extra structural equations.
    # MQA is the day the second marketing touch arrived; SQL is a week before the decision on opportunities.
    mqa_day = np.full(n, np.nan)
    # Filled below from the touch table once concatenated.

    touches = {
        "account_id": np.concatenate(touch_account) if touch_account else np.array([], dtype=int),
        "contact_id": np.concatenate(touch_contact) if touch_contact else np.array([], dtype=int),
        "channel_idx": np.concatenate(touch_channel) if touch_channel else np.array([], dtype=int),
        "day": np.concatenate(touch_day) if touch_day else np.array([], dtype=float),
        "role_weight": np.concatenate(touch_weight) if touch_weight else np.array([], dtype=float),
        "post_decision": np.zeros(0 if not touch_account else sum(len(x) for x in touch_account), dtype=bool),
    }
    if sales_account:
        sa = np.concatenate(sales_account)
        sd = np.concatenate(sales_day)
        sp = np.concatenate(sales_post)
        # Sales carries the economic-buyer weight only as a logged activity, not as a role-weighted cause.
        touches = {
            "account_id": np.concatenate([touches["account_id"], sa]),
            "contact_id": np.concatenate([touches["contact_id"], sa * 10]),  # economic buyer slot 1 if present, else account*10
            "channel_idx": np.concatenate([touches["channel_idx"], np.full(sa.size, IDX["sales_ae"])]),
            "day": np.concatenate([touches["day"], sd]),
            "role_weight": np.concatenate([touches["role_weight"], np.ones(sa.size)]),
            "post_decision": np.concatenate([touches["post_decision"], sp]),
        }
    order = np.lexsort((touches["day"], touches["account_id"]))
    for k in touches:
        touches[k] = touches[k][order]

    # Second touch day = MQA, when it exists.
    acc_t = touches["account_id"]
    day_t = touches["day"]
    mkt_mask = (touches["channel_idx"] < K_MKT)
    acc_m = acc_t[mkt_mask]
    day_m = day_t[mkt_mask]
    if acc_m.size:
        # count within account
        change = np.r_[True, acc_m[1:] != acc_m[:-1]]
        seq = np.cumsum(change)
        # position within account
        starts = np.flatnonzero(change)
        pos = np.arange(acc_m.size) - np.repeat(starts, np.diff(np.r_[starts, acc_m.size]))
        second = pos == 1
        mqa_day[acc_m[second]] = day_m[second]

    sql_day = np.where(y == 1, DECISION_DAY - 7.0, np.nan)
    opp_day = np.where(y == 1, DECISION_DAY, np.nan)
    close_day = np.where(w == 1, 160.0, np.nan)

    return dict(
        n=n, seed=seed, channels=list(CHANNELS), mkt=list(MKT), z_names=list(Z_NAMES),
        segment=segment, icp=icp, intent=intent, existing=existing.astype(bool), z=z, acv=acv,
        n_contacts=n_contacts, e=e, s=s, s_sales=s_sales, agent=agent.astype(bool),
        iab=iab, y=y.astype(bool), w=w.astype(bool), pi=pi, p_opp=p_opp, p_win=p_win,
        ai_executed=ai_executed.astype(bool), hold_abm=hold_abm, hold_llm=hold_llm, gold=gold,
        mqa_day=mqa_day, sql_day=sql_day, opp_day=opp_day, close_day=close_day,
        touches=touches, contacts=contact_rows, coef=true_coef(),
        decision_day=DECISION_DAY, horizon_day=180.0, half_life=HALF_LIFE,
    )
