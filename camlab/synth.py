"""Synthetic data with known ground truth, so every estimate can be scored."""

import numpy as np

from .camera import Camera
from .geometry import look_at, rodrigues_inv


def checkerboard(nx=9, ny=6, square=0.03):
    """Corner positions of a flat target, centred on the origin, in its own frame (z = 0). N x 3."""
    xs, ys = np.meshgrid(np.arange(nx) * square, np.arange(ny) * square)
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny)])
    return pts - pts.mean(axis=0)


def random_board_pose(rng, distance=(0.35, 0.8), max_tilt_deg=50):
    """A board position and tilt that puts it in front of a camera at the origin. Returns rvec, t (board frame to camera frame)."""
    d = rng.uniform(*distance)
    offset = rng.uniform(-0.05, 0.05, 2) * d
    # Tilt the board by a random rotation about a random axis in its plane, then add some spin about the viewing axis.
    tilt = np.radians(rng.uniform(5, max_tilt_deg))
    axis_angle = rng.uniform(0, 2 * np.pi)
    axis = np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0])
    spin = rng.uniform(-np.pi, np.pi)
    from .geometry import rodrigues

    R = rodrigues(axis * tilt) @ rodrigues([0, 0, spin])
    return rodrigues_inv(R), np.array([offset[0], offset[1], d])


def render_views(camera, board, n_views, rng, noise=0.0, image_size=(1280, 960), max_tilt_deg=50):
    """Project the board in several random poses. Returns observations and the true poses. Only views where every corner lands in the image are kept."""
    obs, rvecs, tvecs = [], [], []
    w, h = image_size
    while len(obs) < n_views:
        rv, t = random_board_pose(rng, max_tilt_deg=max_tilt_deg)
        uv = camera.project_world(board, rv, t)
        if not np.all(np.isfinite(uv)) or uv[:, 0].min() < 0 or uv[:, 1].min() < 0 or uv[:, 0].max() > w or uv[:, 1].max() > h:
            continue
        obs.append(uv + rng.normal(0, noise, uv.shape))
        rvecs.append(rv)
        tvecs.append(t)
    return obs, rvecs, tvecs


def reference_camera():
    """A plausible 1280 x 960 camera with some barrel distortion."""
    return Camera(fx=1050.0, fy=1040.0, cx=641.0, cy=478.0, skew=0.0, k1=-0.22, k2=0.07, p1=0.0008, p2=-0.0006)


def two_view_scene(rng, n_points=200, baseline=0.5, rotation_deg=12, depth=(3.0, 8.0), camera=None, noise=0.0, outliers=0.0, image_size=(1280, 960)):
    """Random 3D points seen from two cameras. Returns pixels in both, the true relative pose (camera 1 to camera 2) and the 3D points in camera 1's frame."""
    from .geometry import rodrigues

    cam = camera or Camera(900.0, 900.0, 640.0, 480.0)
    z = rng.uniform(*depth, n_points)
    xy = rng.uniform(-0.45, 0.45, (n_points, 2)) * z[:, None]
    X1 = np.column_stack([xy, z])
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R = rodrigues(axis * np.radians(rotation_deg))
    direction = rng.normal(size=3)
    direction[2] *= 0.3
    direction /= np.linalg.norm(direction)
    t = direction * baseline
    X2 = X1 @ R.T + t
    x1, x2 = cam.project(X1), cam.project(X2)
    keep = np.all(np.isfinite(x2), axis=1) & (x2[:, 0] > 0) & (x2[:, 0] < image_size[0]) & (x2[:, 1] > 0) & (x2[:, 1] < image_size[1])
    X1, x1, x2 = X1[keep], x1[keep], x2[keep]
    x1 = x1 + rng.normal(0, noise, x1.shape)
    x2 = x2 + rng.normal(0, noise, x2.shape)
    bad = rng.random(len(x1)) < outliers
    x2[bad] = rng.uniform([0, 0], image_size, (bad.sum(), 2))
    return cam, x1, x2, R, t, X1, ~bad
