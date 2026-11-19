"""Baseline attribution models. Every function returns per-journey credit rows for converting
journeys (shape [n_conv, K], each row sums to 1) so aggregate share = mean of rows, matching the
benchmark's 'aggregate_true_contribution' convention (each converting journey counts equally)."""
from __future__ import annotations
import numpy as np
from .common import Paths, shapley_matrix, normalize_rows, LN2


def _conv_touch_mask(P: Paths):
    return P.y[P.jidx] == 1


def _rows_from_touch_weights(P: Paths, w: np.ndarray) -> np.ndarray:
    K = len(P.channels)
    conv_rows = np.flatnonzero(P.y == 1)
    pos = -np.ones(P.n, int); pos[conv_rows] = np.arange(len(conv_rows))
    m = _conv_touch_mask(P)
    C = np.zeros((len(conv_rows), K))
    np.add.at(C, (pos[P.jidx[m]], P.ch[m]), w[m])
    return normalize_rows(C)


def _first_last(P: Paths):
    first = np.r_[True, P.jidx[1:] != P.jidx[:-1]]
    last = np.r_[P.jidx[1:] != P.jidx[:-1], True]
    return first, last


def last_click(P):
    _, last = _first_last(P); return _rows_from_touch_weights(P, last.astype(float))


def first_click(P):
    first, _ = _first_last(P); return _rows_from_touch_weights(P, first.astype(float))


def linear(P):
    return _rows_from_touch_weights(P, np.ones(len(P.t)))


def time_decay(P, half_life=168.0, t_end=720.0):
    return _rows_from_touch_weights(P, np.exp(-LN2 / half_life * (t_end - P.t)))


def position_based(P, first_w=0.4, last_w=0.4):
    first, last = _first_last(P)
    n_t = np.bincount(P.jidx, minlength=P.n)[P.jidx]
    mid = (1 - first_w - last_w)
    w = np.where(n_t == 1, 1.0,
        np.where(n_t == 2, np.where(first, 0.5, 0.5),
        np.where(first, first_w, np.where(last, last_w, mid / np.maximum(n_t - 2, 1)))))
    return _rows_from_touch_weights(P, w)


def markov_removal(P):
    """First-order Markov chain: start -> channels -> {conv, null}. Removal effect per channel."""
    K = len(P.channels); S, CONV, NULL = K, K + 1, K + 2
    first, last = _first_last(P)
    T = np.zeros((K + 3, K + 3))
    np.add.at(T, (np.full(first.sum(), S), P.ch[first]), 1)
    nxt = ~last
    np.add.at(T, (P.ch[:-1][nxt[:-1]], P.ch[1:][nxt[:-1]]), 1)
    end_state = np.where(P.y[P.jidx[last]] == 1, CONV, NULL)
    np.add.at(T, (P.ch[last], end_state), 1)
    Q = T / np.maximum(T.sum(1, keepdims=True), 1)

    def p_conv(Q):
        trans = list(range(K)) + [S]
        A = np.eye(len(trans)) - Q[np.ix_(trans, trans)]
        b = Q[trans, CONV]
        x = np.linalg.solve(A, b)
        return x[-1]
    base = p_conv(Q)
    re = np.zeros(K)
    for c in range(K):
        Qr = Q.copy(); Qr[:, NULL] += Qr[:, c]; Qr[:, c] = 0; Qr[c, :] = 0; Qr[c, NULL] = 1
        re[c] = 1 - p_conv(Qr) / base
    # per-journey: split each converting journey's credit across its present channels ∝ removal effect
    conv_rows = np.flatnonzero(P.y == 1)
    pres = P.counts()[conv_rows] > 0
    rows = normalize_rows(pres * re[None, :])
    return rows, re / re.sum()


def _channel_sets(P):
    pres = P.counts() > 0
    code = pres @ (1 << np.arange(len(P.channels)))
    return pres, code.astype(int)


def shapley_zhao_counts(P):
    """Zhao et al. (2018) simplified Shapley with v(S)=sum_{T subset S} conversions(T).
    Closed form: each converting journey's credit is split equally across its distinct channels."""
    pres, _ = _channel_sets(P)
    return normalize_rows(pres[P.y == 1].astype(float))


def shapley_set_cr(P):
    """Shapley per converting journey with value v(S) = empirical conversion rate of journeys whose
    exact channel set is S (v(empty)=0, unseen sets -> 0). Rows normalized (negatives clipped)."""
    K = len(P.channels)
    pres, code = _channel_sets(P)
    n_set = np.bincount(code, minlength=1 << K); c_set = np.bincount(code, weights=P.y, minlength=1 << K)
    cr = np.divide(c_set, n_set, out=np.zeros(1 << K), where=n_set > 0)
    masks, M = shapley_matrix(K)
    mask_code = masks @ (1 << np.arange(K))
    conv_pres = pres[P.y == 1]
    # coalition S is allowed only if subset of journey set; value = cr(S) else equal to cr(S ∩ T)
    sub = (masks[None, :, :] & conv_pres[:, None, :]) @ (1 << np.arange(K))   # code of S∩T
    V = cr[sub]
    phi = np.clip(V @ M, 0, None) * conv_pres
    return normalize_rows(phi)
