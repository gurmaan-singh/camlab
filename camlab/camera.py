"""A pinhole camera with Brown-Conrady lens distortion."""

from dataclasses import dataclass

import numpy as np

from .geometry import rodrigues


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0
    k1: float = 0.0   # radial distortion, r^2 term
    k2: float = 0.0   # radial distortion, r^4 term
    p1: float = 0.0   # tangential distortion
    p2: float = 0.0

    @property
    def K(self):
        return np.array([[self.fx, self.skew, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def distort(self, xn):
        """Apply lens distortion to normalised image coordinates (N x 2)."""
        x, y = xn[:, 0], xn[:, 1]
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2
        xd = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return np.column_stack([xd, yd])

    def undistort(self, xd, iterations=30):
        """Invert `distort` by fixed-point iteration (normalised coordinates, N x 2)."""
        x = xd.copy()
        for _ in range(iterations):
            guess = self.distort(x)
            x = x - (guess - xd)
        return x

    def project(self, X_cam):
        """3D points in the camera frame (N x 3) to pixels (N x 2). Points behind the camera give nan."""
        X_cam = np.asarray(X_cam, float)
        z = X_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = X_cam[:, :2] / z[:, None]
        xn = np.where((z > 1e-9)[:, None], xn, np.nan)
        xd = self.distort(xn)
        u = self.fx * xd[:, 0] + self.skew * xd[:, 1] + self.cx
        v = self.fy * xd[:, 1] + self.cy
        return np.column_stack([u, v])

    def project_world(self, X_world, rvec, t):
        R = rodrigues(rvec)
        return self.project(np.asarray(X_world, float) @ R.T + t)

    def normalise_pixels(self, uv):
        """Pixels to undistorted normalised coordinates."""
        uv = np.asarray(uv, float)
        y = (uv[:, 1] - self.cy) / self.fy
        x = (uv[:, 0] - self.cx - self.skew * y) / self.fx
        return self.undistort(np.column_stack([x, y]))

    def undistort_pixels(self, uv):
        """Remove the lens distortion from pixel coordinates, keeping the same intrinsics."""
        xn = self.normalise_pixels(uv)
        return np.column_stack([self.fx * xn[:, 0] + self.skew * xn[:, 1] + self.cx, self.fy * xn[:, 1] + self.cy])

    def params(self):
        return np.array([self.fx, self.fy, self.cx, self.cy, self.skew, self.k1, self.k2, self.p1, self.p2])

    @staticmethod
    def from_params(p):
        return Camera(*[float(v) for v in p])
