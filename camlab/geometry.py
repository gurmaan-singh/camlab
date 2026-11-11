"""Rotations, homogeneous coordinates and the homography."""

import numpy as np


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rodrigues(rvec):
    """Rotation vector (axis times angle) to rotation matrix."""
    rvec = np.asarray(rvec, float)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3) + skew(rvec)
    k = rvec / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def rodrigues_inv(R):
    """Rotation matrix to rotation vector, stable near 0 and near pi."""
    R = np.asarray(R, float)
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(cos)
    if theta < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if np.pi - theta < 1e-5:
        # Near a half turn the antisymmetric part vanishes: take the axis from the symmetric part instead.
        M = (R + np.eye(3)) / 2
        axis = np.sqrt(np.maximum(np.diag(M), 0))
        i = int(np.argmax(axis))
        axis = M[:, i] / max(axis[i], 1e-12)
        axis /= np.linalg.norm(axis)
        # Fix the sign using whatever antisymmetric part remains.
        s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if s @ axis < 0:
            axis = -axis
        return axis * theta
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return axis * theta


def rotation_angle(R1, R2=None):
    """Angle in radians of the rotation R1 R2^T (or of R1 alone)."""
    R = R1 if R2 is None else R1 @ np.asarray(R2).T
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def project_to_rotation(M):
    """The nearest rotation matrix to M (Frobenius norm), via the SVD."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def look_at(eye, target, up=(0, 0, 1)):
    """World-to-camera rotation and translation for a camera at `eye` looking at `target` (z forward, x right, y down)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ eye


def normalize_points(pts):
    """Hartley normalisation: move the centroid to the origin and scale so the mean distance is sqrt(2). Returns points and the 3x3 transform."""
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    d = np.mean(np.linalg.norm(pts - c, axis=1))
    s = np.sqrt(2) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return (pts - c) * s, T


def to_homogeneous(p):
    p = np.asarray(p, float)
    return np.hstack([p, np.ones((len(p), 1))])


def from_homogeneous(p):
    return p[:, :-1] / p[:, -1:]


def estimate_homography(src, dst):
    """Homography H with dst ~ H src (both N x 2, N >= 4), by the normalised direct linear transform."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4:
        raise ValueError("a homography needs at least 4 point pairs")
    s, Ts = normalize_points(src)
    d, Td = normalize_points(dst)
    A = np.zeros((2 * len(s), 9))
    for i, ((x, y), (u, v)) in enumerate(zip(s, d)):
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    _, sv, Vt = np.linalg.svd(A)
    if sv[-2] < 1e-10 * sv[0]:
        raise ValueError("degenerate configuration: the points are collinear or too few distinct")
    H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def apply_homography(H, pts):
    return from_homogeneous(to_homogeneous(pts) @ H.T)
