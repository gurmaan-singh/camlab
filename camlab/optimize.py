"""Levenberg-Marquardt for nonlinear least squares."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LMResult:
    x: np.ndarray
    cost: float                      # 0.5 * sum of squared residuals
    iterations: int
    converged: bool
    history: list = field(default_factory=list)


def numeric_jacobian(fun, x, h=1e-6):
    f0 = fun(x)
    J = np.empty((len(f0), len(x)))
    for i in range(len(x)):
        step = h * max(1.0, abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] += step
        xm[i] -= step
        J[:, i] = (fun(xp) - fun(xm)) / (2 * step)
    return J


def levenberg_marquardt(fun, x0, max_iter=100, tol=1e-12, jac=None, lam0=1e-3):
    """Minimise 0.5 * |fun(x)|^2. `fun` returns the residual vector. The damping grows after a bad step and shrinks after a good one."""
    x = np.array(x0, float)
    r = fun(x)
    cost = 0.5 * float(r @ r)
    lam = lam0
    history = [cost]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        J = jac(x) if jac else numeric_jacobian(fun, x)
        g = J.T @ r
        A = J.T @ J
        improved = False
        for _ in range(20):
            step = np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-12)), -g)
            r_new = fun(x + step)
            cost_new = 0.5 * float(r_new @ r_new)
            if np.isfinite(cost_new) and cost_new < cost:
                x, r = x + step, r_new
                rel = (cost - cost_new) / max(cost, 1e-300)
                cost = cost_new
                lam = max(lam / 3, 1e-12)
                improved = True
                break
            lam *= 4
        history.append(cost)
        if not improved or rel < tol:
            converged = True
            break
    return LMResult(x, cost, it, converged, history)
