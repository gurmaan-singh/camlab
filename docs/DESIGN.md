# Design notes

## Calibration step by step

A flat target in the plane z = 0 is seen in several views. For one view, the mapping from target coordinates (x, y) to pixels is a homography `H = K [r1 r2 t]`, where `r1` and `r2` are the first two columns of the rotation. Because `r1` and `r2` are orthonormal, two equations on `B = K^-T K^-1` follow for each view:

```
h1' B h2 = 0
h1' B h1 = h2' B h2
```

`B` has six unknowns (symmetric 3 x 3), so three views give a linear system whose null vector is `B`. `K` comes from `B` in closed form (Zhang's formulas), then `r1`, `r2`, `t` for each view from `H` and `K`, with `r3 = r1 x r2` and the result projected onto the nearest true rotation by SVD.

That closed-form answer ignores lens distortion and minimises an algebraic error, not the thing we care about. So it is only a starting point: Levenberg-Marquardt then adjusts `fx, fy, cx, cy`, optionally skew and tangential terms, `k1, k2`, and all the poses, to minimise the pixel distance between the projected and observed corners.

## Conditioning

**Normalising the points.** The direct linear transform for a homography is badly conditioned on raw pixel coordinates (values around 1000 next to 1). Both point sets are first moved to zero mean and scaled to a mean distance of the square root of 2 (Hartley's normalisation), and the result is mapped back. Without it the noiseless homography test did not reach 1e-8. The same applies to the 3D points in the DLT for pose.

**Scaling the homographies.** The linear system for `B` mixes entries of very different sizes. The homographies are first expressed in pixel coordinates divided by the image size, and `K` is scaled back afterwards.

**Telling a good set of views from a degenerate one.** If every view is parallel to every other (the board slides or spins in front of the camera but never tilts), the equations for `B` are nearly dependent, and the solver still returns something: finite, plausible and wrong. I measured the ratio of the second-smallest to the largest singular value of the system: 0.17 to 0.23 for good random views, 5e-4 for views that only slide and spin, 4e-5 for the same view repeated. The code refuses below 5e-3. That threshold sits between the two groups by a large margin on both sides, which is why I trust it; it is not derived from theory.

## Pose from a known object

For a flat object, the same homography trick gives the pose. For a general 3D object it is the direct linear transform on the 3 x 4 projection matrix, which needs at least six points that are not coplanar. The DLT checks the second-smallest singular value, so coplanar points are refused rather than given a meaningless pose. Signs are fixed by requiring the object to be in front of the camera. Both are refined with the full lens model, after first undistorting the pixels.

## Two views

The normalised eight-point algorithm: normalise both point sets, solve for `F` from `x2' F x1 = 0`, force rank 2 by zeroing the smallest singular value, and undo the normalisation. RANSAC draws eight matches at a time, scores every match by its Sampson distance (a first-order approximation of the distance to the epipolar constraint, which is cheap and accurate near the solution), keeps the largest consensus set, and refits on all of it.

The essential matrix is `K2' F K1`, cleaned to have two equal singular values and one zero. It decomposes into four poses (two rotations, two signs of translation). The right one is the one for which triangulated points lie in front of both cameras.

One bug worth noting: the adaptive stopping rule for RANSAC, which works out how many draws are needed given the share of inliers found so far, divided by `log(1 - w^8)`, and when `w^8` was small enough that `1 - w^8` rounded to exactly 1, the logarithm was zero and the program crashed with an overflow. The fix computes it with `log1p` and falls back to the full iteration count when the inlier share is too small to estimate.

## What the figure of the principal point shows

In the calibration-accuracy plot the principal point error flattens at about 3 pixels while the focal length keeps improving. I thought it was a bug until I traced it: the synthetic camera has tangential distortion and the default calibration does not fit it, so a fixed bias remains. With tangential terms fitted the error is 0.6 px at the same noise. The plot is kept as it is, because it is a useful demonstration that the answer can be wrong in a way that more data does not fix.

## Not done

- Feature detection and matching, and anything that touches a real image.
- Bundle adjustment over many views and points, and an analytic Jacobian for it.
- Fisheye and rolling-shutter models.
