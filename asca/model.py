"""ASCA v0 - Agentic Structural Causal Attribution.

Structural causal model (per journey j, channel c):
  Exposure kernel:   E_jc(h) = sum_{touches of c} exp(-ln2/h (T_end - t))          (adstock, half-life h)
  Saturation:        s_jc     = E_jc / (kappa + E_jc)                                (Hill, n=1)
  Agent node:        A_j ~ Bernoulli(pi_j),  logit pi_j = a0 + sum_c g_c s_jc + zA'Z_j
  Outcome:           Y_j ~ Bernoulli(p_jA),  logit p_ja = b0 + sum_c b_c s_jc + d*a + zY'Z_j
  Agent proxy (opt): M_j | Y_j=1, A_j ~ Bernoulli(q*A_j)    (e.g. observed 'agent_checkout'),
                     M is a *descendant* of A and Y, never used as an exposure.
A is latent and marginalised in the likelihood. Z are observed pre-treatment covariates (backdoor set).

Inference: MAP with weakly-informative Gaussian priors, Laplace approximation (finite-difference
Hessian) -> posterior draws of all parameters incl. log h, log kappa.

Credit: for each converting journey, v_j(S) = E[Y | do(X_S = x_jS, X_{-S} = 0), Z_j]
        = (1-pi_j(S)) p_j0(S) + pi_j(S) p_j1(S); phi_j = exact Shapley of v_j over channels;
        rows normalized to 1 and averaged (benchmark convention); incremental conversions =
        sum_j [v_j(all) - v_j(empty)] over all journeys, allocated by un-normalized Shapley.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from .common import Paths, shapley_matrix, normalize_rows


class ASCA:
    def __init__(self, K, n_z=0, latent_agent=True, proxy=False, prior_sd=2.0, t_end=720.0,
                 h_prior=(np.log(100.0), 2.0), kappa_prior=(0.0, 2.0), fixed_shape=None):
        self.K, self.nz, self.la, self.proxy, self.sd, self.t_end = K, n_z, latent_agent, proxy, prior_sd, t_end
        self.h_prior, self.k_prior, self.fixed_shape = h_prior, kappa_prior, fixed_shape
        names = ["b0"] + [f"b_{c}" for c in range(K)] + [f"zY_{i}" for i in range(n_z)]
        if latent_agent:
            names += ["d", "a0"] + [f"g_{c}" for c in range(K)] + [f"zA_{i}" for i in range(n_z)]
            if proxy:
                names += ["logit_q"]
        if fixed_shape is None:
            names += ["log_h", "log_kappa"]
        self.names = names

    # ---------- parameter unpacking ----------
    def unpack(self, th):
        K, nz = self.K, self.nz; i = 0
        p = {}
        p["b0"] = th[i]; i += 1
        p["b"] = th[i:i + K]; i += K
        p["zY"] = th[i:i + nz]; i += nz
        if self.la:
            p["d"] = th[i]; p["a0"] = th[i + 1]; i += 2
            p["g"] = th[i:i + K]; i += K
            p["zA"] = th[i:i + nz]; i += nz
            if self.proxy:
                p["lq"] = th[i]; i += 1
        if self.fixed_shape is None:
            p["log_h"], p["log_kappa"] = th[i], th[i + 1]
        else:
            p["log_h"], p["log_kappa"] = np.log(self.fixed_shape[0]), np.log(self.fixed_shape[1])
        return p

    def sat(self, p, E):
        k = np.exp(p["log_kappa"])
        return E / (k + E)

    # ---------- likelihood ----------
    def _exposure(self, p):
        h = np.exp(p["log_h"])
        w = np.exp(-np.log(2) / h * (self.t_end - self._t))
        E = np.zeros((self._n, self.K)); np.add.at(E, (self._j, self._c), w)
        return E

    def neg_log_post(self, th):
        p = self.unpack(th)
        s = self.sat(p, self._exposure(p))
        Z = self._Z
        eta = p["b0"] + s @ p["b"] + (Z @ p["zY"] if self.nz else 0)
        y = self._y
        if not self.la:
            ll = np.sum(y * log_expit(eta) + (1 - y) * log_expit(-eta))
        else:
            lp = p["a0"] + s @ p["g"] + (Z @ p["zA"] if self.nz else 0)
            l_pi1, l_pi0 = log_expit(lp), log_expit(-lp)
            e1 = eta + p["d"]
            if not self.proxy:
                l1 = np.logaddexp(l_pi0 + log_expit(eta), l_pi1 + log_expit(e1))
                l0 = np.logaddexp(l_pi0 + log_expit(-eta), l_pi1 + log_expit(-e1))
                ll = np.sum(np.where(y == 1, l1, l0))
            else:
                lq1, lq0 = log_expit(p["lq"]), log_expit(-p["lq"])
                m = self._m
                l0 = np.logaddexp(l_pi0 + log_expit(-eta), l_pi1 + log_expit(-e1))
                l1m = l_pi1 + log_expit(e1) + lq1
                l1n = np.logaddexp(l_pi0 + log_expit(eta), l_pi1 + log_expit(e1) + lq0)
                ll = np.sum(np.where(y == 0, l0, np.where(m == 1, l1m, l1n)))
        # priors
        lpr = -0.5 * (p["b0"] / 5) ** 2 - 0.5 * np.sum((p["b"] / self.sd) ** 2) - 0.5 * np.sum((p["zY"] / self.sd) ** 2)
        if self.la:
            lpr += -0.5 * (p["d"] / self.sd) ** 2 - 0.5 * (p["a0"] / 5) ** 2 - 0.5 * np.sum((p["g"] / self.sd) ** 2) \
                   - 0.5 * np.sum((p["zA"] / self.sd) ** 2)
            if self.proxy:
                lpr += -0.5 * (p["lq"] / 2.5) ** 2
        if self.fixed_shape is None:
            lpr += -0.5 * ((p["log_h"] - self.h_prior[0]) / self.h_prior[1]) ** 2 \
                   - 0.5 * ((p["log_kappa"] - self.k_prior[0]) / self.k_prior[1]) ** 2
        return -(ll + lpr)

    # ---------- fitting ----------
    def fit(self, P: Paths, Z=None, M=None, seed=0, n_starts=2, verbose=True):
        self._j, self._c, self._t, self._n, self._y = P.jidx, P.ch, P.t, P.n, P.y.astype(float)
        self._Z = np.zeros((P.n, 0)) if Z is None else np.asarray(Z, float)
        self._m = None if M is None else np.asarray(M, float)
        rng = np.random.default_rng(seed)
        best = None
        for s in range(n_starts):
            th0 = np.zeros(len(self.names)); th0[0] = np.log(P.y.mean() / (1 - P.y.mean()))
            th0 += rng.normal(0, 0.1, len(th0)) * (s > 0)
            if self.la:
                th0[self.names.index("d")] = 1.0 + 0.5 * s; th0[self.names.index("a0")] = -1.5
            if self.fixed_shape is None:
                th0[self.names.index("log_h")] = np.log(100.0)
            r = minimize(self.neg_log_post, th0, method="L-BFGS-B", options=dict(maxiter=3000, maxfun=200000))
            if verbose:
                print(f"  start {s}: nlp={r.fun:.2f} success={r.success} nit={r.nit}")
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x; self.opt = best
        self.H = self._hessian(best.x)
        try:
            self.cov = np.linalg.inv(self.H)
            w, V = np.linalg.eigh((self.cov + self.cov.T) / 2)
            self.cov = (V * np.clip(w, 1e-10, None)) @ V.T
        except np.linalg.LinAlgError:
            self.cov = np.linalg.pinv(self.H)
        return self

    def _hessian(self, th, rel=1e-4):
        n = len(th); H = np.zeros((n, n)); e = rel * np.maximum(1, np.abs(th)); f0 = self.neg_log_post(th)
        for i in range(n):
            for k in range(i, n):
                if i == k:
                    tp = th.copy(); tp[i] += e[i]; tm = th.copy(); tm[i] -= e[i]
                    H[i, i] = (self.neg_log_post(tp) - 2 * f0 + self.neg_log_post(tm)) / e[i] ** 2
                else:
                    v = []
                    for si, sk in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        t = th.copy(); t[i] += si * e[i]; t[k] += sk * e[k]; v.append(self.neg_log_post(t))
                    H[i, k] = H[k, i] = (v[0] - v[1] - v[2] + v[3]) / (4 * e[i] * e[k])
        return H

    def draws(self, n, seed=1):
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self.theta, self.cov, size=n)

    def summary(self):
        sd = np.sqrt(np.diag(self.cov))
        return {nm: (self.theta[i], sd[i]) for i, nm in enumerate(self.names)}

    # ---------- interventional credit ----------
    def coalition_values(self, th, P: Paths, rows, Z=None, block_agent_path=False, block_direct_path=False):
        """V[j, S] = E[Y | do(X_S = x_jS, X_-S = 0), Z_j] for journeys `rows`.
        block_agent_path: A-equation held at X=0 (only direct X->Y path active).
        block_direct_path: Y-equation's X held at 0 (only X->A->Y path active)."""
        p = self.unpack(th)
        h = np.exp(p["log_h"])
        w = np.exp(-np.log(2) / h * (self.t_end - P.t))
        E = np.zeros((P.n, self.K)); np.add.at(E, (P.jidx, P.ch), w)
        s = self.sat(p, E)[rows]
        masks, Mshap = shapley_matrix(self.K)
        Zr = np.zeros((len(rows), 0)) if Z is None else np.asarray(Z, float)[rows]
        cY = (s * p["b"]) @ masks.T.astype(float)                     # [n, 2^K]
        if block_direct_path:
            cY = np.zeros_like(cY)
        eta = p["b0"] + cY + ((Zr @ p["zY"])[:, None] if self.nz else 0)
        if not self.la:
            V = expit(eta)
        else:
            cA = (s * p["g"]) @ masks.T.astype(float)
            if block_agent_path:
                cA = np.zeros_like(cA)
            lp = p["a0"] + cA + ((Zr @ p["zA"])[:, None] if self.nz else 0)
            pi = expit(lp)
            V = (1 - pi) * expit(eta) + pi * expit(eta + p["d"])
        return V, Mshap

    def credit(self, th, P, Z=None, **kw):
        rows = np.flatnonzero(P.y == 1)
        V, Mshap = self.coalition_values(th, P, rows, Z, **kw)
        phi = V @ Mshap                                                  # [n_conv, K]
        return phi

    def incremental(self, th, P, Z=None, batch=20000):
        """Total model-implied incremental conversions over ALL journeys, by channel (Shapley)."""
        tot = np.zeros(self.K)
        for st in range(0, P.n, batch):
            rows = np.arange(st, min(st + batch, P.n))
            V, Mshap = self.coalition_values(th, P, rows, Z)
            tot += (V @ Mshap).sum(0)
        return tot
