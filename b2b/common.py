"""Channel list, Shapley weights, and share scoring for the B2B benchmark."""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.stats import kendalltau, spearmanr

from asca.common import shapley_matrix

LN2 = np.log(2.0)

# Marketing channels, then post-opportunity sales. Order is the column order everywhere.
CHANNELS = [
    "paid_search",
    "linkedin",
    "content",
    "events",
    "sdr",
    "partner",
    "review",
    "organic",
    "plg",
    "abm",
    "llm_research",
    "wom",
    "podcast",
    "sales_ae",
]
MKT = CHANNELS[:-1]
SALES = "sales_ae"
AI = "ai_sourced"  # observed, non-causal. Truth share is 0. Not in the SCM exposures.
SCORE_CHANNELS = CHANNELS + [AI]

K_MKT = len(MKT)  # 13
K_ALL = len(CHANNELS)  # 14

IDX = {c: i for i, c in enumerate(CHANNELS)}
AGENT_CHANNELS = ["content", "review", "llm_research", "podcast"]
AGENT_IDX = np.array([IDX[c] for c in AGENT_CHANNELS], dtype=int)

# Touch-level measurement class. Floors keep CRM channels visible when cookies die.
CHANNEL_CLASS = {
    "paid_search": "cookie",
    "linkedin": "cookie",
    "content": "cookie",
    "events": "offline",
    "sdr": "crm",
    "partner": "crm",
    "review": "cookie",
    "organic": "cookie",
    "plg": "crm",
    "abm": "cookie",
    "llm_research": "dark",
    "wom": "dark",
    "podcast": "offline",
    "sales_ae": "crm",
}
CLASS_FLOOR = {"cookie": 0.08, "offline": 0.30, "crm": 0.80, "dark": 0.0}
# Visibility when tracking quality q = 1 (a careful stack still misses dark and cookieless touches).
BASE_VISIBILITY = np.array([
    0.82,  # paid_search
    0.20,  # linkedin touch log (account-level impressions are separate)
    0.62,  # content
    0.48,  # events
    0.97,  # sdr
    0.72,  # partner
    0.34,  # review
    0.50,  # organic
    0.93,  # plg
    0.18,  # abm touch log
    0.03,  # llm
    0.07,  # wom
    0.16,  # podcast
    0.96,  # sales
])
AGGREGATE_CHANNELS = ("linkedin", "abm")
EXPERIMENT_CHANNELS = ("llm_research", "abm")
TOUCH_ID_THRESHOLD = 0.15

DECISION_DAY = 110.0
HORIZON_DAY = 180.0
HALF_LIFE = 45.0
KAPPA = 1.0

Z_NAMES = ["icp", "intent", "existing", "mid_market", "enterprise"]

QUALITIES = {
    "near_perfect": 1.00,
    "good": 0.72,
    "degraded": 0.42,
    "heavy": 0.18,
}

ROLES = ["champion", "economic_buyer", "tech_evaluator", "end_user"]
ROLE_WEIGHT = {"champion": 1.15, "economic_buyer": 1.30, "tech_evaluator": 0.80, "end_user": 0.40}


def visibility(q: float) -> np.ndarray:
    """Per-channel probability a true touch survives into the production log."""
    floors = np.array([CLASS_FLOOR[CHANNEL_CLASS[c]] for c in CHANNELS])
    return BASE_VISIBILITY * (floors + (1.0 - floors) * q)


_MASKS = None
_SHAPLEY_M = None


def masks_and_weights(k: int = K_ALL):
    global _MASKS, _SHAPLEY_M
    if _MASKS is None or _MASKS.shape[1] != k:
        _MASKS, _SHAPLEY_M = shapley_matrix(k)
        _MASKS = _MASKS.astype(np.float64)
    return _MASKS, _SHAPLEY_M


def saturate(e: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    return e / (kappa + e)


def mix_prob(eta: np.ndarray, d: float, logit_pi: np.ndarray) -> np.ndarray:
    """P(Y=1) with a binary latent agent marginalized out."""
    pi = expit(logit_pi)
    return (1.0 - pi) * expit(eta) + pi * expit(eta + d)


def outcome_values(s_mkt: np.ndarray, s_sales: np.ndarray, z: np.ndarray, acv: np.ndarray,
                   coef: dict, outcome: str, batch: int = 2000, opp_offset: np.ndarray | None = None) -> np.ndarray:
    """Per-account Shapley of expected dollars. s_mkt is (N, 13) saturated marketing exposure.

    Pipeline value is ACV * P(opportunity). Revenue value is ACV * P(opportunity) * P(win | opportunity).
    Sales exposure is included in a coalition only when the sales column of the coalition is on.
    The opportunity equation does not depend on sales, so sales' pipeline Shapley is zero.
    """
    masks, shap_m = masks_and_weights(K_ALL)
    masks = np.asarray(masks, dtype=np.float32)
    shap_m = np.asarray(shap_m, dtype=np.float32)
    n = s_mkt.shape[0]
    s_mkt = np.asarray(s_mkt, dtype=np.float32)
    s_sales = np.asarray(s_sales, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    acv = np.asarray(acv, dtype=np.float32)
    b = np.zeros(K_ALL)
    b[:K_MKT] = coef["b_opp"]
    # sales coefficient on opportunity is structurally zero
    g = np.zeros(K_MKT)
    g[AGENT_IDX] = coef["g"]
    phi = np.zeros((n, K_ALL))
    z_opp = z @ coef["z_opp"]
    z_a = z @ coef["z_agent"]
    if outcome == "revenue":
        c = np.zeros(K_ALL)
        c[:K_MKT] = coef["b_win"]
        c[IDX[SALES]] = coef["delta_sales"]
        z_win = z @ coef["z_win"]
    for start in range(0, n, batch):
        sl = slice(start, min(start + batch, n))
        sm = s_mkt[sl]
        # Coalition marketing dose: (batch, 2^K)
        dose_b = (sm * b[:K_MKT]) @ masks[:, :K_MKT].T
        dose_g = (sm * g) @ masks[:, :K_MKT].T
        eta = coef["b0_opp"] + dose_b + z_opp[sl, None]
        if opp_offset is not None:
            eta = eta + opp_offset[sl, None]
        lp = coef["a0"] + dose_g + z_a[sl, None]
        p_opp = mix_prob(eta, coef["d_opp"], lp)
        if outcome == "pipeline":
            v = acv[sl, None] * p_opp
        else:
            sales_on = s_sales[sl, None] * masks[:, IDX[SALES]][None, :]
            dose_c = (sm * c[:K_MKT]) @ masks[:, :K_MKT].T + c[IDX[SALES]] * sales_on
            eta_w = coef["b0_win"] + dose_c + z_win[sl, None]
            p_win = mix_prob(eta_w, coef["d_win"], lp)
            v = acv[sl, None] * p_opp * p_win
        phi[sl] = v @ shap_m
    return phi


def shares_from_phi(phi: np.ndarray) -> np.ndarray:
    total = phi.sum()
    if not np.isfinite(total) or abs(total) < 1e-12:
        return np.zeros(phi.shape[1])
    return phi.sum(0) / total


def rank_scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    with np.errstate(invalid="ignore"):
        sp = spearmanr(pred, truth).statistic
        kd = kendalltau(pred, truth).statistic
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MaxAE": float(np.max(np.abs(err))),
        "Spearman": float(sp) if np.isfinite(sp) else float("nan"),
        "Kendall": float(kd) if np.isfinite(kd) else float("nan"),
    }


def coverage(lo: np.ndarray, hi: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((lo <= truth) & (truth <= hi)))
