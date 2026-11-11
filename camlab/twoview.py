"""Two views of the same scene: the fundamental and essential matrices, relative pose, triangulation."""

import numpy as np

from .geometry import from_homogeneous, normalize_points, project_to_rotation, to_homogeneous


def fundamental_8pt(x1, x2):
    """Fundamental matrix F with x2' F x1 = 0 from at least 8 correspondences (N x 2 pixels), by the normalised eight-point algorithm."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    if len(x1) < 8:
        raise ValueError("the eight-point algorithm needs at least 8 correspondences")
    p1, T1 = normalize_points(x1)
    p2, T2 = normalize_points(x2)
    A = np.column_stack([p2[:, 0] * p1[:, 0], p2[:, 0] * p1[:, 1], p2[:, 0], p2[:, 1] * p1[:, 0], p2[:, 1] * p1[:, 1], p2[:, 1], p1[:, 0], p1[:, 1], np.ones(len(p1))])
    _, sv, Vt = np.linalg.svd(A)
    if sv[-2] < 1e-9 * sv[0]:
        raise ValueError("degenerate configuration: the points do not determine a fundamental matrix")
    F = Vt[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)          # a fundamental matrix has rank 2
    F = U @ np.diag([S[0], S[1], 0]) @ Vt
    F = T2.T @ F @ T1
    return F / np.linalg.norm(F)


def sampson_distance(F, x1, x2):
    """First-order approximation to the squared distance (pixels squared) of each correspondence from satisfying the epipolar constraint."""
    h1, h2 = to_homogeneous(x1), to_homogeneous(x2)
    Fx1 = h1 @ F.T
    Ftx2 = h2 @ F
    num = np.sum(h2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / np.maximum(den, 1e-18)


def ransac_fundamental(x1, x2, threshold=1.0, iterations=1000, rng=None, confidence=0.999):
    """Robust fundamental matrix. Returns F and a boolean inlier mask. The number of iterations adapts to the inlier share found so far."""
    rng = rng or np.random.default_rng(0)
    n = len(x1)
    if n < 8:
        raise ValueError("need at least 8 correspondences")
    best_mask, best_count = None, -1
    needed = iterations
    it = 0
    while it < min(iterations, needed):
        it += 1
        idx = rng.choice(n, 8, replace=False)
        try:
            F = fundamental_8pt(x1[idx], x2[idx])
        except ValueError:
            continue
        mask = sampson_distance(F, x1, x2) < threshold**2
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
            # How many samples are needed to draw an all-inlier set of 8 with the given confidence, given the inlier share so far.
            p_good = min((count / n) ** 8, 1 - 1e-12)
            needed = iterations if p_good < 1e-9 else int(np.ceil(np.log(1 - confidence) / np.log1p(-p_good)))
    if best_mask is None or best_count < 8:
        raise ValueError("no consistent model found")
    F = fundamental_8pt(x1[best_mask], x2[best_mask])   # refit on all inliers
    return F, sampson_distance(F, x1, x2) < threshold**2


def essential_from_fundamental(F, K1, K2=None):
    E = (K2 if K2 is not None else K1).T @ F @ K1
    U, S, Vt = np.linalg.svd(E)
    s = (S[0] + S[1]) / 2   # a true essential matrix has two equal singular values and a zero
    return U @ np.diag([s, s, 0]) @ Vt


def decompose_essential(E):
    """The four (R, t) pairs consistent with E. t is a unit vector; its length cannot be recovered from two views."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1] *= -1
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    t = U[:, 2]
    return [(U @ W @ Vt, t), (U @ W @ Vt, -t), (U @ W.T @ Vt, t), (U @ W.T @ Vt, -t)]


def triangulate(P1, P2, x1, x2):
    """Linear (DLT) triangulation of N point pairs given two 3 x 4 projection matrices and points in the same normalised or pixel frame. Returns N x 3."""
    out = np.empty((len(x1), 3))
    for i, (a, b) in enumerate(zip(x1, x2)):
        A = np.array([a[0] * P1[2] - P1[0], a[1] * P1[2] - P1[1], b[0] * P2[2] - P2[0], b[1] * P2[2] - P2[1]])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        out[i] = X[:3] / X[3]
    return out


def recover_pose(E, xn1, xn2):
    """The relative pose (R, unit t) that puts the most triangulated points in front of both cameras. xn are normalised coordinates."""
    best, best_count = None, -1
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    for R, t in decompose_essential(E):
        P2 = np.hstack([R, t[:, None]])
        X = triangulate(P1, P2, xn1, xn2)
        in_front1 = X[:, 2] > 0
        in_front2 = (X @ R.T + t)[:, 2] > 0
        count = int(np.sum(in_front1 & in_front2))
        if count > best_count:
            best, best_count = (R, t), count
    return best[0], best[1], best_count


def epipolar_lines(F, x1):
    """The line in image 2 (a, b, c with ax + by + c = 0) on which each point of image 1 must lie."""
    return to_homogeneous(x1) @ F.T
