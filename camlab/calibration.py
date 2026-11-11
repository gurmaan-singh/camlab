"""Camera calibration from several views of a flat target (Zhang's method), with nonlinear refinement."""

from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .geometry import estimate_homography, project_to_rotation, rodrigues, rodrigues_inv
from .optimize import levenberg_marquardt


@dataclass
class CalibrationResult:
    camera: Camera
    rvecs: list
    tvecs: list
    rms_initial: float       # reprojection error (pixels) of the closed-form solution
    rms: float               # after refinement
    iterations: int


def _v(H, i, j):
    h = H.T
    return np.array([
        h[i, 0] * h[j, 0],
        h[i, 0] * h[j, 1] + h[i, 1] * h[j, 0],
        h[i, 1] * h[j, 1],
        h[i, 2] * h[j, 0] + h[i, 0] * h[j, 2],
        h[i, 2] * h[j, 1] + h[i, 1] * h[j, 2],
        h[i, 2] * h[j, 2],
    ])


def intrinsics_from_homographies(Hs):
    """Closed-form K from the homographies of a plane, as in Zhang 2000. Needs at least three views with different orientations."""
    if len(Hs) < 3:
        raise ValueError("need at least 3 views to solve for all of fx, fy, cx, cy and skew")
    rows = []
    for H in Hs:
        rows.append(_v(H, 0, 1))
        rows.append(_v(H, 0, 0) - _v(H, 1, 1))
    _, sv, Vt = np.linalg.svd(np.array(rows))
    if sv[-2] < 5e-3 * sv[0]:
        # Several independent directions fit equally well: the views do not pin the intrinsics down. Typical for
        # views that are all parallel to each other, such as a board that only slides or spins in front of the camera.
        raise ValueError("degenerate views: the target must be seen from several clearly different orientations")
    b = Vt[-1]
    if b[0] < 0:
        b = -b
    B11, B12, B22, B13, B23, B33 = b
    denom = B11 * B22 - B12**2
    if abs(denom) < 1e-18:
        raise ValueError("degenerate views: the target was not seen from sufficiently different directions")
    v0 = (B12 * B13 - B11 * B23) / denom
    lam = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    if lam / B11 <= 0 or lam * B11 / denom <= 0:
        raise ValueError("degenerate views: no valid intrinsic matrix (try views from more different angles)")
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / denom)
    gamma = -B12 * alpha**2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha**2 / lam
    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1.0]])


def extrinsics_from_homography(K, H):
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 1.0 / np.linalg.norm(Kinv @ h1)
    r1, r2 = lam * Kinv @ h1, lam * Kinv @ h2
    t = lam * Kinv @ h3
    R = project_to_rotation(np.column_stack([r1, r2, np.cross(r1, r2)]))
    if t[2] < 0:   # the target must be in front of the camera
        R = project_to_rotation(np.column_stack([-r1, -r2, np.cross(r1, r2)]))
        t = -t
    return R, t


def reprojection_rms(camera, board, observations, rvecs, tvecs):
    err = []
    for pts, rv, t in zip(observations, rvecs, tvecs):
        err.append((camera.project_world(board, rv, t) - pts).ravel())
    e = np.concatenate(err)
    return float(np.sqrt(np.mean(e**2) * 2))   # root-mean-square distance in pixels, not per coordinate


def calibrate(board, observations, fit_skew=False, fit_tangential=False, image_size=None, refine=True):
    """Calibrate from views of a planar target.

    board: N x 3 target points in the target's own frame (z = 0).
    observations: list of N x 2 arrays, the pixel position of each target point in each view.
    """
    board = np.asarray(board, float)
    if np.ptp(board[:, 2]) > 1e-9:
        raise ValueError("calibrate() expects a planar target (all z equal)")
    Hs = [estimate_homography(board[:, :2], o) for o in observations]
    # Work in roughly unit pixel coordinates so the linear system is well conditioned, then undo it.
    s = float(np.max(np.abs(np.concatenate(observations)))) if image_size is None else float(max(image_size))
    S = np.diag([1 / s, 1 / s, 1.0])
    K = np.linalg.inv(S) @ intrinsics_from_homographies([S @ H for H in Hs])
    K[2] = [0, 0, 1]
    ext = [extrinsics_from_homography(K, H) for H in Hs]
    rvecs = [rodrigues_inv(R) for R, _ in ext]
    tvecs = [t for _, t in ext]
    cam = Camera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1] if fit_skew else 0.0)
    rms0 = reprojection_rms(cam, board, observations, rvecs, tvecs)
    if not refine:
        return CalibrationResult(cam, rvecs, tvecs, rms0, rms0, 0)

    n = len(observations)
    free_cam = [0, 1, 2, 3] + ([4] if fit_skew else []) + [5, 6] + ([7, 8] if fit_tangential else [])

    def unpack(x):
        p = np.zeros(9)
        p[free_cam] = x[: len(free_cam)]
        cam_ = Camera.from_params(p)
        views = x[len(free_cam):].reshape(n, 6)
        return cam_, views[:, :3], views[:, 3:]

    def residuals(x):
        cam_, rv, tv = unpack(x)
        return np.concatenate([(cam_.project_world(board, rv[i], tv[i]) - observations[i]).ravel() for i in range(n)])

    x0 = np.concatenate([cam.params()[free_cam], np.hstack([np.array(rvecs), np.array(tvecs)]).ravel()])
    res = levenberg_marquardt(residuals, x0, max_iter=60)
    cam_, rv, tv = unpack(res.x)
    return CalibrationResult(cam_, list(rv), list(tv), rms0, reprojection_rms(cam_, board, observations, list(rv), list(tv)), res.iterations)
