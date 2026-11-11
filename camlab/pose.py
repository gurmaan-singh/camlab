"""Where is the camera? Pose from known 3D points and their pixels (Perspective-n-Point)."""

import numpy as np

from .camera import Camera
from .geometry import estimate_homography, normalize_points, project_to_rotation, rodrigues, rodrigues_inv
from .optimize import levenberg_marquardt


def pose_from_dlt(points, xn):
    """Pose from at least 6 points not all on a plane, by the direct linear transform on the 3 x 4 projection matrix."""
    X = np.asarray(points, float)
    n = len(X)
    if n < 6:
        raise ValueError("the DLT needs at least 6 points")
    # Centre and scale the 3D points first: raw coordinates make the linear system badly conditioned.
    c = X.mean(axis=0)
    sc = np.sqrt(3) / max(np.mean(np.linalg.norm(X - c, axis=1)), 1e-12)
    T = np.array([[sc, 0, 0, -sc * c[0]], [0, sc, 0, -sc * c[1]], [0, 0, sc, -sc * c[2]], [0, 0, 0, 1.0]])
    Xh = np.hstack([(X - c) * sc, np.ones((n, 1))])
    A = np.zeros((2 * n, 12))
    for i in range(n):
        u, v = xn[i]
        A[2 * i, 0:4] = Xh[i]
        A[2 * i, 8:12] = -u * Xh[i]
        A[2 * i + 1, 4:8] = Xh[i]
        A[2 * i + 1, 8:12] = -v * Xh[i]
    _, sv, Vt = np.linalg.svd(A)
    if sv[-2] < 1e-9 * sv[0]:
        raise ValueError("degenerate configuration for the DLT: the points are coplanar or too few")
    P = Vt[-1].reshape(3, 4) @ T
    M = P[:, :3]
    if np.linalg.det(M) < 0:
        P = -P
        M = P[:, :3]
    s = np.cbrt(np.linalg.det(M))
    P = P / s
    R = project_to_rotation(P[:, :3])
    t = P[:, 3]
    if (R @ X.mean(axis=0) + t)[2] < 0:
        R, t = project_to_rotation(-P[:, :3]), -t
    return R, t


def solve_pnp(points, pixels, camera: Camera, refine=True):
    """Camera pose (rvec, t) that maps world points into the camera frame, from at least 4 coplanar or 6 general points.

    The pixels are corrected for lens distortion first, a linear solution gives the start, and Levenberg-Marquardt
    minimises the reprojection error through the full camera model.
    """
    points = np.asarray(points, float)
    pixels = np.asarray(pixels, float)
    if len(points) != len(pixels):
        raise ValueError("points and pixels must match")
    xn = camera.normalise_pixels(pixels)
    planar = np.ptp(points[:, 2]) < 1e-9 * max(1.0, np.abs(points).max())
    if planar:
        H = estimate_homography(points[:, :2], xn)
        lam = 1.0 / np.linalg.norm(H[:, 0])
        r1, r2, t = lam * H[:, 0], lam * H[:, 1], lam * H[:, 2]
        R = project_to_rotation(np.column_stack([r1, r2, np.cross(r1, r2)]))
        if t[2] < 0:
            R = project_to_rotation(np.column_stack([-r1, -r2, np.cross(r1, r2)]))
            t = -t
        t = t + R[:, 2] * points[0, 2]
    else:
        R, t = pose_from_dlt(points, xn)
    x0 = np.concatenate([rodrigues_inv(R), t])
    if not refine:
        return x0[:3], x0[3:]

    def residuals(x):
        return (camera.project_world(points, x[:3], x[3:]) - pixels).ravel()

    res = levenberg_marquardt(residuals, x0, max_iter=50)
    return res.x[:3], res.x[3:]
