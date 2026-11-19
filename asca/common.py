"""Shared data structures, Shapley utilities and scoring metrics."""
from __future__ import annotations
import itertools
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

LN2 = np.log(2.0)


@dataclass
class Paths:
    """Flat touchpoint representation, sorted by (journey, time)."""
    jidx: np.ndarray      # journey row index (0..n-1) per touch
    ch: np.ndarray        # channel index per touch
    t: np.ndarray         # touch time (hours)
    y: np.ndarray         # outcome per journey (0/1)
    channels: list        # channel names
    n: int

    @classmethod
    def from_frames(cls, tp: pd.DataFrame, journey_ids: np.ndarray, y: np.ndarray, channels: list):
        pos = pd.Series(np.arange(len(journey_ids)), index=journey_ids)
        tp = tp[tp.journey_id.isin(journey_ids)].sort_values(["journey_id", "t_hours", "step"] if "step" in tp else ["journey_id", "t_hours"])
        cmap = {c: i for i, c in enumerate(channels)}
        return cls(pos.loc[tp.journey_id.values].values, tp.channel.map(cmap).values.astype(int),
                   tp.t_hours.values.astype(float), np.asarray(y).astype(int), list(channels), len(journey_ids))

    def subset(self, rows: np.ndarray) -> "Paths":
        rows = np.asarray(rows)
        newpos = -np.ones(self.n, dtype=int)
        # allow repeated rows (bootstrap): expand touch table
        order = np.argsort(self.jidx, kind="stable")
        starts = np.searchsorted(self.jidx[order], np.arange(self.n))
        ends = np.searchsorted(self.jidx[order], np.arange(self.n), side="right")
        lens = ends[rows] - starts[rows]
        tidx = np.concatenate([order[s:e] for s, e in zip(starts[rows], ends[rows])]) if len(rows) else np.array([], int)
        new_j = np.repeat(np.arange(len(rows)), lens)
        return Paths(new_j, self.ch[tidx], self.t[tidx], self.y[rows], self.channels, len(rows))

    def decayed_exposure(self, half_life: float, t_end: float = 720.0, K: int | None = None) -> np.ndarray:
        """E[j,c] = sum over touches of c in journey j of exp(-ln2/h * (t_end - t))."""
        K = K or len(self.channels)
        w = np.exp(-LN2 / half_life * (t_end - self.t))
        E = np.zeros((self.n, K))
        np.add.at(E, (self.jidx, self.ch), w)
        return E

    def counts(self, K: int | None = None) -> np.ndarray:
        K = K or len(self.channels)
        N = np.zeros((self.n, K))
        np.add.at(N, (self.jidx, self.ch), 1.0)
        return N


def shapley_matrix(K: int):
    """Return (masks [2^K,K] bool, M [2^K,K]) so that phi = V @ M for coalition values V[...,2^K].

    phi_c = sum_{S not containing c} |S|!(K-|S|-1)!/K! (v(S+c) - v(S)).
    """
    masks = np.array(list(itertools.product([0, 1], repeat=K)), dtype=bool)[:, ::-1]
    idx = {tuple(m): i for i, m in enumerate(masks)}
    from math import factorial
    M = np.zeros((len(masks), K))
    for i, m in enumerate(masks):
        s = m.sum()
        for c in range(K):
            if not m[c]:
                w = factorial(s) * factorial(K - s - 1) / factorial(K)
                m2 = m.copy(); m2[c] = True
                M[idx[tuple(m2)], c] += w
                M[i, c] -= w
    return masks, M


def normalize_rows(C: np.ndarray) -> np.ndarray:
    s = C.sum(1, keepdims=True)
    out = np.divide(C, s, out=np.zeros_like(C), where=s > 0)
    return out


def score(pred: dict, truth: dict, channels: list) -> dict:
    p = np.array([pred.get(c, 0.0) for c in channels]); g = np.array([truth[c] for c in channels])
    err = p - g
    return dict(MAE=np.abs(err).mean(), MaxAE=np.abs(err).max(),
                Spearman=spearmanr(p, g).statistic, Kendall=kendalltau(p, g).statistic)


def per_journey_mae(pred_rows: np.ndarray, gt_rows: np.ndarray) -> float:
    """Mean over converting journeys of mean-over-channels |share_pred - share_gt| (both row-normalized)."""
    return float(np.abs(normalize_rows(pred_rows) - normalize_rows(gt_rows)).mean())
