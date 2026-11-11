"""Run the experiments and draw the figures for the README.

    python scripts/make_figures.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from camlab.calibration import calibrate
from camlab.geometry import rotation_angle
from camlab.synth import checkerboard, reference_camera, render_views, two_view_scene
from camlab.twoview import epipolar_lines, essential_from_fundamental, ransac_fundamental, recover_pose

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
BG, FG, DIM, RED = "#070707", "#e8e8e8", "#7a7a7a", "#ff2d2d"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": "#333", "axes.labelcolor": FG,
    "text.color": FG, "xtick.color": DIM, "ytick.color": DIM, "grid.color": "#1c1c1c", "axes.grid": True, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def calibration_accuracy():
    cam, board = reference_camera(), checkerboard()
    views = [3, 5, 8, 12, 20]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    results = {}
    for noise, colour in ((0.1, FG), (0.3, "#b0b0b0"), (1.0, RED)):
        fe, pe = [], []
        for n in views:
            f, p = [], []
            for seed in range(6):
                obs, _, _ = render_views(cam, board, n, np.random.default_rng(1000 * n + seed), noise=noise)
                c = calibrate(board, obs).camera
                f.append(100 * np.hypot(c.fx - cam.fx, c.fy - cam.fy) / cam.fx)
                p.append(np.hypot(c.cx - cam.cx, c.cy - cam.cy))
            fe.append(np.mean(f))
            pe.append(np.mean(p))
        results[noise] = (fe, pe)
        ax[0].plot(views, fe, color=colour, lw=1.6, marker="o", ms=4, label=f"{noise} px noise")
        ax[1].plot(views, pe, color=colour, lw=1.6, marker="o", ms=4, label=f"{noise} px noise")
        print("noise", noise, "focal error %", np.round(fe, 3), "principal point px", np.round(pe, 2))
    ax[0].set(xlabel="number of views of the board", ylabel="focal length error (%)", title="Focal length", yscale="log")
    ax[1].set(xlabel="number of views of the board", ylabel="principal point error (pixels)", title="Principal point", yscale="log")
    ax[0].legend()
    save(fig, "calibration_accuracy.png")


def distortion_figure():
    cam, board = reference_camera(), checkerboard()
    obs, _, _ = render_views(cam, board, 15, np.random.default_rng(3), noise=0.3)
    res = calibrate(board, obs, fit_tangential=True)
    est = res.camera
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # A regular grid of normalised coordinates as seen through the true lens and as corrected by the estimated one.
    gx, gy = np.meshgrid(np.linspace(-0.55, 0.55, 12), np.linspace(-0.42, 0.42, 9))
    xn = np.column_stack([gx.ravel(), gy.ravel()])
    true_px = np.column_stack([cam.fx * cam.distort(xn)[:, 0] + cam.cx, cam.fy * cam.distort(xn)[:, 1] + cam.cy])
    fixed = est.undistort_pixels(true_px)
    ideal = np.column_stack([cam.fx * xn[:, 0] + cam.cx, cam.fy * xn[:, 1] + cam.cy])
    for a, pts, title in ((ax[0], true_px, "What the lens does to a square grid"), (ax[1], fixed, "After undoing the estimated distortion")):
        g = pts.reshape(9, 12, 2)
        for i in range(9):
            a.plot(g[i, :, 0], g[i, :, 1], color=RED, lw=1.2)
        for j in range(12):
            a.plot(g[:, j, 0], g[:, j, 1], color=RED, lw=1.2)
        i0 = ideal.reshape(9, 12, 2)
        for i in range(9):
            a.plot(i0[i, :, 0], i0[i, :, 1], color=DIM, lw=0.6, ls="--")
        for j in range(12):
            a.plot(i0[:, j, 0], i0[:, j, 1], color=DIM, lw=0.6, ls="--")
        a.set(title=title, aspect="equal", xlim=(0, 1280), ylim=(960, 0), xticks=[], yticks=[])
        a.grid(False)
    err = np.linalg.norm(fixed - ideal, axis=1)
    ax[1].text(20, 940, f"largest remaining error {err.max():.2f} px (grey dashed: the ideal grid)", color=DIM)
    save(fig, "distortion.png")


def epipolar_figure():
    rng = np.random.default_rng(11)
    cam, x1, x2, R, t, X, inliers = two_view_scene(rng, n_points=120, noise=0.5, outliers=0.3)
    F, mask = ransac_fundamental(x1, x2, threshold=1.5, rng=np.random.default_rng(0))
    E = essential_from_fundamental(F, cam.K)
    Rr, tr, _ = recover_pose(E, cam.normalise_pixels(x1[mask]), cam.normalise_pixels(x2[mask]))
    rot_err = np.degrees(rotation_angle(Rr, R))
    dir_err = np.degrees(np.arccos(np.clip(tr @ (t / np.linalg.norm(t)), -1, 1)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, pts, title in ((ax[0], x1, "View 1"), (ax[1], x2, "View 2")):
        a.scatter(pts[mask, 0], pts[mask, 1], s=10, color=FG, label="kept by RANSAC")
        a.scatter(pts[~mask, 0], pts[~mask, 1], s=14, color=RED, marker="x", label="rejected")
        a.set(title=title, xlim=(0, 1280), ylim=(960, 0), aspect="equal", xticks=[], yticks=[])
        a.grid(False)
    lines = epipolar_lines(F, x1[mask][:12])
    for (a_, b_, c_), col in zip(lines, np.linspace(0.3, 0.9, 12)):
        xs = np.array([0, 1280])
        ax[1].plot(xs, -(a_ * xs + c_) / b_, color=RED, lw=0.7, alpha=0.7)
    ax[0].legend(loc="lower left")
    ax[1].text(20, 940, f"relative rotation error {rot_err:.2f} deg, translation direction error {dir_err:.2f} deg", color=DIM)
    print("epipolar example: inliers kept", mask.sum(), "of", len(mask), "rot err", round(rot_err, 3), "dir err", round(dir_err, 3))
    save(fig, "epipolar.png")


def two_view_accuracy():
    noises = [0.0, 0.25, 0.5, 1.0, 2.0]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for outliers, colour in ((0.0, FG), (0.3, RED)):
        errs = []
        for noise in noises:
            e = []
            for seed in range(8):
                cam, x1, x2, R, t, X, _ = two_view_scene(np.random.default_rng(50 + seed), n_points=250, noise=noise, outliers=outliers)
                F, mask = ransac_fundamental(x1, x2, threshold=max(1.0, 3 * noise), rng=np.random.default_rng(seed))
                Rr, tr, _ = recover_pose(essential_from_fundamental(F, cam.K), cam.normalise_pixels(x1[mask]), cam.normalise_pixels(x2[mask]))
                e.append(np.degrees(rotation_angle(Rr, R)))
            errs.append(np.median(e))
        ax.plot(noises, errs, color=colour, lw=1.6, marker="o", ms=4, label=f"{int(outliers * 100)} % wrong matches")
    ax.set(xlabel="pixel noise (sigma)", ylabel="median rotation error (degrees)", title="Relative rotation from two views, 250 points, 12 degree turn")
    ax.legend()
    save(fig, "two_view_accuracy.png")


if __name__ == "__main__":
    distortion_figure()
    epipolar_figure()
    two_view_accuracy()
    calibration_accuracy()
