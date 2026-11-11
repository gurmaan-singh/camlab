import numpy as np
import pytest

from camlab.calibration import calibrate, intrinsics_from_homographies
from camlab.camera import Camera
from camlab.geometry import (apply_homography, estimate_homography, look_at, project_to_rotation, rodrigues, rodrigues_inv, rotation_angle)
from camlab.optimize import levenberg_marquardt
from camlab.pose import pose_from_dlt, solve_pnp
from camlab.synth import checkerboard, reference_camera, render_views, two_view_scene
from camlab.twoview import (decompose_essential, epipolar_lines, essential_from_fundamental, fundamental_8pt, ransac_fundamental, recover_pose, sampson_distance, triangulate)


@pytest.fixture(scope="module")
def board():
    return checkerboard()


# ---------------------------------------------------------------- rotations and homographies
def test_rodrigues_round_trip_including_awkward_angles():
    rng = np.random.default_rng(0)
    for angle in [0.0, 1e-8, 0.3, 1.5, 3.0, np.pi - 1e-7, np.pi]:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rv = axis * angle
        R = rodrigues(rv)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
        back = rodrigues(rodrigues_inv(R))
        assert rotation_angle(back, R) < 1e-6, angle


def test_nearest_rotation_projection():
    rng = np.random.default_rng(1)
    R = rodrigues(rng.normal(size=3))
    noisy = R + rng.normal(0, 0.01, (3, 3))
    P = project_to_rotation(noisy)
    assert np.allclose(P @ P.T, np.eye(3), atol=1e-12) and np.linalg.det(P) > 0
    assert rotation_angle(P, R) < 0.05
    assert np.linalg.det(project_to_rotation(-np.eye(3))) > 0   # a reflection becomes a proper rotation


def test_look_at_points_the_camera_at_the_target():
    R, t = look_at([1, 2, 3], [0, 0, 0])
    target_in_cam = R @ np.zeros(3) + t
    assert target_in_cam[0] == pytest.approx(0, abs=1e-12) and target_in_cam[1] == pytest.approx(0, abs=1e-12) and target_in_cam[2] > 0
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) > 0


def test_homography_is_recovered_exactly_without_noise():
    rng = np.random.default_rng(2)
    H = np.array([[1.1, 0.2, 30.0], [-0.1, 0.9, -10.0], [1e-4, -2e-4, 1.0]])
    src = rng.uniform(0, 100, (12, 2))
    dst = apply_homography(H, src)
    est = estimate_homography(src, dst)
    assert np.allclose(est, H / H[2, 2], atol=1e-8)


def test_homography_estimation_refuses_bad_input():
    with pytest.raises(ValueError, match="at least 4"):
        estimate_homography(np.zeros((3, 2)), np.zeros((3, 2)))
    line = np.column_stack([np.arange(8.0), 2 * np.arange(8.0)])
    with pytest.raises(ValueError, match="degenerate"):
        estimate_homography(line, line + 1)


# ---------------------------------------------------------------- the camera model
def test_distortion_is_invertible():
    cam = reference_camera()
    rng = np.random.default_rng(3)
    xn = rng.uniform(-0.5, 0.5, (500, 2))
    assert np.max(np.abs(cam.undistort(cam.distort(xn)) - xn)) < 1e-9


def test_projection_and_back_projection_agree():
    cam = reference_camera()
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.uniform(-0.3, 0.3, (200, 2)), rng.uniform(0.5, 2.0, 200)])
    uv = cam.project(X)
    xn = cam.normalise_pixels(uv)
    assert np.allclose(xn, X[:, :2] / X[:, 2:], atol=1e-8)


def test_points_behind_the_camera_do_not_project():
    cam = Camera(800, 800, 320, 240)
    assert np.all(np.isnan(cam.project(np.array([[0.1, 0.1, -1.0]]))))
    assert cam.K[0, 0] == 800 and cam.K[2, 2] == 1


def test_skew_enters_the_projection():
    cam = Camera(800, 800, 320, 240, skew=5.0)
    uv = cam.project(np.array([[0.0, 0.1, 1.0]]))[0]
    assert uv[0] == pytest.approx(320 + 5.0 * 0.1) and uv[1] == pytest.approx(240 + 80)


# ---------------------------------------------------------------- calibration
def test_noiseless_calibration_recovers_everything_exactly(board):
    cam = reference_camera()
    obs, _, _ = render_views(cam, board, 12, np.random.default_rng(5))
    res = calibrate(board, obs, fit_tangential=True)
    assert np.max(np.abs(res.camera.params() - cam.params())[:4]) < 1e-4
    assert abs(res.camera.k1 - cam.k1) < 1e-6 and abs(res.camera.k2 - cam.k2) < 1e-6
    assert abs(res.camera.p1 - cam.p1) < 1e-6 and abs(res.camera.p2 - cam.p2) < 1e-6
    assert res.rms < 1e-4


def test_closed_form_solution_is_good_enough_to_start_from(board):
    cam = Camera(1050.0, 1040.0, 641.0, 478.0)   # no distortion: the linear step alone should be very close
    obs, _, _ = render_views(cam, board, 10, np.random.default_rng(6))
    res = calibrate(board, obs, refine=False)
    assert abs(res.camera.fx - 1050) < 1e-3 and abs(res.camera.cy - 478) < 1e-3


def test_with_pixel_noise_the_focal_length_is_accurate_and_the_residual_matches_the_noise(board):
    cam = reference_camera()
    sigma = 0.3
    errs_f, rms = [], []
    for seed in range(6):
        obs, _, _ = render_views(cam, board, 15, np.random.default_rng(100 + seed), noise=sigma)
        res = calibrate(board, obs)
        errs_f.append(abs(res.camera.fx - cam.fx) / cam.fx)
        rms.append(res.rms)
    assert np.mean(errs_f) < 0.005                                   # focal length within half a percent
    assert np.mean(rms) == pytest.approx(sigma * np.sqrt(2), rel=0.15)  # residual distance ~ noise (two coordinates)


def test_refinement_improves_on_the_linear_solution(board):
    cam = reference_camera()
    obs, _, _ = render_views(cam, board, 12, np.random.default_rng(7), noise=0.3)
    res = calibrate(board, obs)
    assert res.rms < res.rms_initial


def test_more_views_give_a_better_answer(board):
    cam = reference_camera()

    def mean_error(n_views):
        out = []
        for seed in range(8):
            obs, _, _ = render_views(cam, board, n_views, np.random.default_rng(200 + seed), noise=0.5)
            c = calibrate(board, obs).camera
            out.append(np.hypot(c.fx - cam.fx, c.fy - cam.fy))
        return np.mean(out)

    assert mean_error(25) < mean_error(5)


def test_skew_can_be_estimated(board):
    cam = Camera(1000.0, 1000.0, 640.0, 480.0, skew=8.0)
    obs, _, _ = render_views(cam, board, 15, np.random.default_rng(8))
    res = calibrate(board, obs, fit_skew=True)
    assert res.camera.skew == pytest.approx(8.0, abs=1e-3)


def test_calibration_rejects_what_cannot_work(board):
    cam = reference_camera()
    obs, _, _ = render_views(cam, board, 2, np.random.default_rng(9))
    with pytest.raises(ValueError, match="at least 3 views"):
        calibrate(board, obs)
    obs, _, _ = render_views(cam, board, 5, np.random.default_rng(9))
    with pytest.raises(ValueError, match="planar"):
        calibrate(np.column_stack([board[:, :2], np.linspace(0, 0.1, len(board))]), obs)
    same = [obs[0] + np.random.default_rng(i).normal(0, 0.01, obs[0].shape) for i in range(4)]   # the same view four times
    with pytest.raises(ValueError, match="degenerate"):
        calibrate(board, same)
    # Views that only slide and spin in front of the camera, never tilting, cannot fix the intrinsics either.
    flat = [cam.project_world(board, rodrigues_inv(rodrigues([0, 0, 0.5 * k])), np.array([0.02 * k, -0.01 * k, 0.5 + 0.05 * k])) for k in range(5)]
    with pytest.raises(ValueError, match="degenerate"):
        calibrate(board, flat)


# ---------------------------------------------------------------- pose from a known object
def test_planar_pose_estimation(board):
    cam = reference_camera()
    obs, rv, tv = render_views(cam, board, 20, np.random.default_rng(10), noise=0.5)
    rot, trans = [], []
    for o, r, t in zip(obs, rv, tv):
        r2, t2 = solve_pnp(board, o, cam)
        rot.append(np.degrees(rotation_angle(rodrigues(r2), rodrigues(r))))
        trans.append(np.linalg.norm(t2 - t) * 1000)
    assert np.mean(rot) < 0.3 and np.max(rot) < 1.0
    assert np.mean(trans) < 1.0   # millimetres, for a board half a metre away


def test_general_pose_estimation_and_refinement_helps():
    cam = reference_camera()
    rng = np.random.default_rng(11)
    pts = np.column_stack([rng.uniform(-0.3, 0.3, (40, 2)), rng.uniform(-0.2, 0.2, 40)])
    rvec, t = np.array([0.1, -0.2, 0.3]), np.array([0.05, -0.02, 1.5])
    uv = cam.project_world(pts, rvec, t) + rng.normal(0, 0.5, (40, 2))
    r_lin, t_lin = solve_pnp(pts, uv, cam, refine=False)
    r_ref, t_ref = solve_pnp(pts, uv, cam)
    e_lin = rotation_angle(rodrigues(r_lin), rodrigues(rvec))
    e_ref = rotation_angle(rodrigues(r_ref), rodrigues(rvec))
    assert np.degrees(e_ref) < 0.1 and e_ref < e_lin
    assert np.linalg.norm(t_ref - t) < 0.002


def test_the_dlt_refuses_coplanar_points():
    pts = np.column_stack([np.random.default_rng(12).uniform(-1, 1, (20, 2)), np.zeros(20)])
    with pytest.raises(ValueError, match="degenerate"):
        pose_from_dlt(pts, np.random.default_rng(13).uniform(-0.3, 0.3, (20, 2)))


def test_estimated_poses_put_the_object_in_front_of_the_camera(board):
    cam = reference_camera()
    obs, _, _ = render_views(cam, board, 10, np.random.default_rng(14))
    for o in obs:
        r, t = solve_pnp(board, o, cam)
        assert (rodrigues(r) @ board.T).T[:, 2].mean() + t[2] > 0


# ---------------------------------------------------------------- two views
def test_eight_point_satisfies_the_epipolar_constraint_exactly_without_noise():
    cam, x1, x2, R, t, X, _ = two_view_scene(np.random.default_rng(15), noise=0.0)
    F = fundamental_8pt(x1, x2)
    assert np.linalg.matrix_rank(F, tol=1e-9) == 2
    assert np.max(sampson_distance(F, x1, x2)) < 1e-14
    lines = epipolar_lines(F, x1)
    d = np.abs(np.sum(lines * np.column_stack([x2, np.ones(len(x2))]), axis=1)) / np.hypot(lines[:, 0], lines[:, 1])
    assert d.max() < 1e-6   # every match lies on its epipolar line, to a millionth of a pixel


def test_essential_matrix_decomposition_contains_the_true_pose():
    for seed in range(10):
        cam, x1, x2, R, t, X, _ = two_view_scene(np.random.default_rng(300 + seed), noise=0.0)
        E = essential_from_fundamental(fundamental_8pt(x1, x2), cam.K)
        td = t / np.linalg.norm(t)
        assert any(rotation_angle(Rc, R) < 1e-5 and np.linalg.norm(tc - td) < 1e-5 for Rc, tc in decompose_essential(E)), seed
        Rr, tr, count = recover_pose(E, cam.normalise_pixels(x1), cam.normalise_pixels(x2))
        assert rotation_angle(Rr, R) < 1e-5 and np.linalg.norm(tr - td) < 1e-5 and count == len(x1)


def test_ransac_survives_forty_percent_outliers():
    cam, x1, x2, R, t, X, inliers = two_view_scene(np.random.default_rng(16), n_points=300, noise=0.5, outliers=0.4)
    F, mask = ransac_fundamental(x1, x2, threshold=1.5, rng=np.random.default_rng(0))
    assert (mask & inliers).sum() / mask.sum() > 0.95        # almost nothing it accepts is wrong
    assert (mask & inliers).sum() / inliers.sum() > 0.8      # and it keeps most of what is right
    E = essential_from_fundamental(F, cam.K)
    Rr, tr, _ = recover_pose(E, cam.normalise_pixels(x1[mask]), cam.normalise_pixels(x2[mask]))
    assert np.degrees(rotation_angle(Rr, R)) < 1.0
    assert np.degrees(np.arccos(np.clip(tr @ (t / np.linalg.norm(t)), -1, 1))) < 3.0


def test_triangulation_recovers_depth_up_to_the_unknown_scale():
    cam, x1, x2, R, t, X, _ = two_view_scene(np.random.default_rng(17), noise=0.0)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t[:, None]])
    Xr = triangulate(P1, P2, cam.normalise_pixels(x1), cam.normalise_pixels(x2))
    assert np.max(np.abs(Xr - X)) < 1e-6
    # With a unit-length translation instead of the true one, everything shrinks by the same factor: depth is only known up to scale.
    td = t / np.linalg.norm(t)
    Xu = triangulate(P1, np.hstack([R, td[:, None]]), cam.normalise_pixels(x1), cam.normalise_pixels(x2))
    assert np.allclose(Xu * np.linalg.norm(t), X, atol=1e-6)


def test_the_eight_point_algorithm_needs_eight_points():
    with pytest.raises(ValueError, match="at least 8"):
        fundamental_8pt(np.zeros((7, 2)), np.zeros((7, 2)))


# ---------------------------------------------------------------- the optimiser
def test_levenberg_marquardt_solves_rosenbrock_and_a_curve_fit():
    rosen = lambda x: np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])
    res = levenberg_marquardt(rosen, [-1.2, 1.0])
    assert np.allclose(res.x, [1, 1], atol=1e-6) and res.cost < 1e-12
    t = np.linspace(0, 3, 40)
    truth = np.array([2.5, 1.3, 0.4])
    data = truth[0] * np.exp(-truth[1] * t) + truth[2]
    res = levenberg_marquardt(lambda p: p[0] * np.exp(-p[1] * t) + p[2] - data, [1.0, 1.0, 0.0])
    assert np.allclose(res.x, truth, atol=1e-6)
    assert all(b <= a + 1e-15 for a, b in zip(res.history, res.history[1:]))   # the cost never goes up
