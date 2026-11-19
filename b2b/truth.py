"""Exact interventional Shapley shares under the B2B structural model."""
from __future__ import annotations

import numpy as np

from b2b.common import SCORE_CHANNELS, outcome_values, shares_from_phi


def ground_truth(world: dict) -> dict:
    coef = world["coef"]
    phi_pipe = outcome_values(world["s"], world["s_sales"], world["z"], world["acv"], coef, "pipeline")
    phi_rev = outcome_values(world["s"], world["s_sales"], world["z"], world["acv"], coef, "revenue")
    # ai_sourced is not in the SCM. Its causal share is zero.
    def pack(phi):
        shares = shares_from_phi(phi)
        out = {c: float(shares[i]) for i, c in enumerate(world["channels"])}
        out["ai_sourced"] = 0.0
        return out, shares

    pipe, pipe_v = pack(phi_pipe)
    rev, rev_v = pack(phi_rev)
    # Sales has a zero column in the opportunity equation, so this is a sanity check, not a new target.
    sales_pipe = pipe["sales_ae"]
    return dict(
        pipeline=pipe, revenue=rev,
        phi_pipeline=phi_pipe, phi_revenue=phi_rev,
        incremental_pipeline_dollars=float(phi_pipe.sum()),
        incremental_revenue_dollars=float(phi_rev.sum()),
        sales_pipeline_share=float(sales_pipe),
        channels=list(SCORE_CHANNELS),
        vector_pipeline=np.r_[pipe_v, 0.0],
        vector_revenue=np.r_[rev_v, 0.0],
    )
