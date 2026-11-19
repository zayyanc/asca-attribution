"""B2B path rules on the production log only. None of these see the gold panel,
the holdouts, firmographics, or the true exposures."""
from __future__ import annotations

import numpy as np

from asca.baselines import markov_removal
from asca.common import Paths

from b2b.common import (
    CHANNELS,
    DECISION_DAY,
    HALF_LIFE,
    HORIZON_DAY,
    LN2,
    SCORE_CHANNELS,
)

_INDEX = {c: i for i, c in enumerate(SCORE_CHANNELS)}


def _events(world, obs, outcome: str):
    touches = obs["touches"]
    seen = obs["seen"]
    day = touches["day"][seen]
    keep = day < DECISION_DAY if outcome == "pipeline" else np.ones(day.size, dtype=bool)
    acc = obs["obs_account"][seen][keep]
    names = np.array(CHANNELS, dtype=object)[touches["channel_idx"][seen][keep]]
    day = day[keep]
    flag = obs["ai_flag"] & (obs["y_obs"] if outcome == "pipeline" else obs["w_obs"])
    if flag.any():
        acc = np.concatenate([acc, np.flatnonzero(flag)])
        names = np.concatenate([names, np.full(int(flag.sum()), "ai_sourced", dtype=object)])
        day = np.concatenate([day, np.full(int(flag.sum()), DECISION_DAY - 1.0)])
    return acc, names, day


def _dollar_share(credit: np.ndarray, acv: np.ndarray, positive: np.ndarray):
    weights = np.where(positive, acv, 0.0)
    num = (credit * weights[:, None]).sum(axis=0)
    denom = num.sum()
    share = num / denom if denom > 0 else np.zeros(num.shape[0])
    dollars = weights.sum()
    missed = weights[credit.sum(1) <= 1e-12].sum() / dollars if dollars > 0 else 0.0
    return share, float(missed)


def _lead_source(obs, positive, n):
    credit = np.zeros((n, len(SCORE_CHANNELS)))
    lead = obs["lead_source"]
    for i, name in enumerate(lead):
        if positive[i] and name in _INDEX:
            credit[i, _INDEX[name]] = 1.0
    return credit


def _path_rules(acc, names, day, positive, mqa, outcome: str, n: int):
    """Linear, time-decay, last touch, W-shaped, Z-shaped. Credit is per positive account."""
    credit = {k: np.zeros((n, len(SCORE_CHANNELS))) for k in
              ("linear", "time_decay", "last_touch", "w_shaped", "z_shaped")}
    if acc.size == 0:
        return credit
    order = np.lexsort((day, acc))
    acc, names, day = acc[order], names[order], day[order]
    change = np.r_[True, acc[1:] != acc[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], acc.size]
    t_end = DECISION_DAY if outcome == "pipeline" else HORIZON_DAY
    for s, e in zip(starts, ends):
        a = int(acc[s])
        if not positive[a]:
            continue
        dd = day[s:e]
        nn = names[s:e]
        m = e - s
        first_i = int(np.argmin(dd))
        last_i = int(np.argmax(dd))
        pre = np.flatnonzero(dd < DECISION_DAY)
        opp_i = int(pre[np.argmax(dd[pre])]) if pre.size else last_i
        if np.isfinite(mqa[a]):
            lead_i = int(np.argmin(np.abs(dd - mqa[a])))
        else:
            lead_i = first_i
        decay = np.exp(-LN2 / HALF_LIFE * (t_end - dd))

        def dump(scheme_w):
            scheme_w = np.asarray(scheme_w, dtype=float)
            total = scheme_w.sum()
            if total <= 0:
                return
            scheme_w = scheme_w / total
            row = credit_row
            for w, name in zip(scheme_w, nn):
                if name in _INDEX and w:
                    row[_INDEX[name]] += w

        # linear
        credit_row = credit["linear"][a]
        dump(np.ones(m))
        credit_row = credit["time_decay"][a]
        dump(decay)
        credit_row = credit["last_touch"][a]
        w = np.zeros(m)
        w[last_i if outcome == "revenue" else opp_i] = 1.0
        dump(w)

        rest_w = np.zeros(m)
        w_shape = np.zeros(m)
        for idx, wt in ((first_i, 0.30), (lead_i, 0.30), (opp_i, 0.30)):
            w_shape[idx] += wt
        milestone = {first_i, lead_i, opp_i}
        others = [i for i in range(m) if i not in milestone]
        if others:
            for i in others:
                w_shape[i] += 0.10 / len(others)
        credit_row = credit["w_shaped"][a]
        dump(w_shape)

        z_shape = np.zeros(m)
        for idx, wt in ((first_i, 0.20), (lead_i, 0.20), (opp_i, 0.20), (last_i, 0.20)):
            z_shape[idx] += wt
        milestone = {first_i, lead_i, opp_i, last_i}
        others = [i for i in range(m) if i not in milestone]
        if others:
            for i in others:
                z_shape[i] += 0.20 / len(others)
        credit_row = credit["z_shaped"][a]
        dump(z_shape)
        del rest_w
    return credit


def _set_cr_chunked(paths: Paths, chunk: int = 64) -> np.ndarray:
    """Same estimator as asca.baselines.shapley_set_cr, without a converters-by-coalitions tensor."""
    from asca.common import normalize_rows, shapley_matrix

    k = len(paths.channels)
    pres = paths.counts() > 0
    code = (pres @ (1 << np.arange(k))).astype(int)
    n_set = np.bincount(code, minlength=1 << k)
    c_set = np.bincount(code, weights=paths.y, minlength=1 << k)
    cr = np.divide(c_set, n_set, out=np.zeros(1 << k), where=n_set > 0)
    masks, shap_m = shapley_matrix(k)
    powers = 1 << np.arange(k)
    conv_pres = pres[paths.y == 1]
    pieces = []
    for start in range(0, conv_pres.shape[0], chunk):
        block = conv_pres[start:start + chunk]
        sub = (masks[None, :, :] & block[:, None, :]) @ powers
        pieces.append(np.clip(cr[sub] @ shap_m, 0, None) * block)
    phi = np.vstack(pieces) if pieces else np.zeros((0, k))
    return normalize_rows(phi)


def _markov_and_set(acc, names, day, positive, n):
    out = {}
    if acc.size == 0:
        z = np.zeros((n, len(SCORE_CHANNELS)))
        return {"markov": z, "set_cr": z.copy()}
    order = np.lexsort((day, acc))
    acc, names, day = acc[order], names[order], day[order]
    ch = np.array([_INDEX[nm] for nm in names], dtype=int)
    y = positive.astype(int)
    paths = Paths(acc.astype(int), ch, day.astype(float), y, list(SCORE_CHANNELS), n)
    rows, _ = markov_removal(paths)
    conv = np.flatnonzero(y == 1)
    credit_m = np.zeros((n, len(SCORE_CHANNELS)))
    credit_m[conv] = rows
    rows_s = _set_cr_chunked(paths)
    credit_s = np.zeros((n, len(SCORE_CHANNELS)))
    credit_s[conv] = rows_s
    out["markov"] = credit_m
    out["set_cr"] = credit_s
    return out


def baseline_credits(world, obs, outcome: str) -> dict[str, np.ndarray]:
    positive = obs["y_obs"] if outcome == "pipeline" else obs["w_obs"]
    acc, names, day = _events(world, obs, outcome)
    credits = _path_rules(acc, names, day, positive, obs["mqa_crm"], outcome, world["n"])
    credits["lead_source"] = _lead_source(obs, positive, world["n"])
    credits.update(_markov_and_set(acc, names, day, positive, world["n"]))
    return credits


def baseline_shares(world, obs, outcome: str) -> dict[str, dict]:
    positive = obs["y_obs"] if outcome == "pipeline" else obs["w_obs"]
    credits = baseline_credits(world, obs, outcome)
    packed = {}
    for name, credit in credits.items():
        share, missed = _dollar_share(credit, world["acv"], positive)
        packed[name] = {"share": share, "unattributed_dollar_fraction": missed, "credit": credit}
    return packed
