import time
import numpy as np
from skimage.feature import canny as sk_canny, corner_harris as sk_corner_harris, corner_peaks as sk_corner_peaks
from skimage.transform import hough_line as sk_hough_line, hough_line_peaks as sk_hough_line_peaks
from skimage.transform import hough_circle as sk_hough_circle, hough_circle_peaks as sk_hough_circle_peaks

from config import KERNELS
from core.math_ops import manual_convolve2d, manual_gaussian_filter, normalize
from core.edge_detection import canny_scratch

def _gray_to_rgb_uint8(gray: np.ndarray) -> np.ndarray:
    """Convert float [0,1] grayscale to uint8 RGB."""
    g = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)

def _draw_line_parametric(rgb: np.ndarray, theta: float, r: float, H: int, W: int, color=(0, 255, 0)) -> None:
    a = np.cos(theta); b = np.sin(theta)
    t = np.linspace(-max(H, W) * 1.5, max(H, W) * 1.5, int(max(H, W) * 3))
    xs = (a * r + W / 2.0 + t * (-b)).astype(int)
    ys = (b * r + H / 2.0 + t *   a ).astype(int)
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    rgb[ys[valid], xs[valid]] = color

def _draw_circle_on_rgb(rgb: np.ndarray, cx: int, cy: int, r: float, H: int, W: int, color=(0, 255, 0)) -> None:
    n_pts = max(8, int(2 * np.pi * r))
    t = np.linspace(0, 2 * np.pi, n_pts + 1)
    xs = (cx + r * np.cos(t)).astype(int)
    ys = (cy + r * np.sin(t)).astype(int)
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(0,0)]:   # 1-px thickness
        xs2 = xs + dx; ys2 = ys + dy
        v2 = valid & (xs2 >= 0) & (xs2 < W) & (ys2 >= 0) & (ys2 < H)
        rgb[ys2[v2], xs2[v2]] = color

def _mark_corners_on_rgb(rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), dot_r: int = 2) -> None:
    ys, xs = np.nonzero(mask)
    H, W = rgb.shape[:2]
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - dot_r), min(H, y + dot_r + 1)
        x0, x1 = max(0, x - dot_r), min(W, x + dot_r + 1)
        rgb[y0:y1, x0:x1] = color

def harris_corner_scratch(gray: np.ndarray, alpha: float = 0.05, sigma: float = 1.0, threshold: float = 1e-4) -> dict:
    t0 = time.perf_counter()
    Ix = manual_convolve2d(gray, KERNELS["Sobel"]["Gx"])
    Iy = manual_convolve2d(gray, KERNELS["Sobel"]["Gy"])
    Ixx, Iyy, Ixy = Ix * Ix, Iy * Iy, Ix * Iy
    
    A = manual_gaussian_filter(Ixx, sigma)
    B = manual_gaussian_filter(Iyy, sigma)
    C = manual_gaussian_filter(Ixy, sigma)
    
    detM  = A * B - C ** 2
    trace = A + B
    Q = detM - alpha * (trace ** 2)
    
    Q_pos_max = Q[Q > 0].max() if (Q > 0).any() else 1.0
    corners = Q > (threshold * Q_pos_max)
    
    overlay = _gray_to_rgb_uint8(gray)
    _mark_corners_on_rgb(overlay, corners, color=(255, 50, 50), dot_r=2)
    
    return dict(
        Ix=normalize(np.abs(Ix)), Iy=normalize(np.abs(Iy)),
        A=normalize(A), B=normalize(B), C=normalize(np.abs(C)),
        detM=normalize(np.maximum(detM, 0)), trace=normalize(trace),
        Q_map=normalize(np.maximum(Q, 0)), corners_mask=corners.astype(np.float64),
        overlay=overlay, elapsed=(time.perf_counter()-t0)*1000, n_corners=int(corners.sum()),
    )

def harris_corner_library(gray: np.ndarray, alpha: float = 0.05, sigma: float = 1.0) -> dict:
    t0 = time.perf_counter()
    response = sk_corner_harris(gray, method='k', k=alpha, sigma=sigma)
    coords   = sk_corner_peaks(response, min_distance=5, threshold_rel=0.1)
    
    overlay = _gray_to_rgb_uint8(gray)
    for r, c in coords:
        y0, y1 = max(0, r-2), min(gray.shape[0], r+3)
        x0, x1 = max(0, c-2), min(gray.shape[1], c+3)
        overlay[y0:y1, x0:x1] = (50, 220, 50)
        
    return dict(response=normalize(np.maximum(response, 0)), overlay=overlay, n_corners=len(coords), elapsed=(time.perf_counter()-t0)*1000)

def hough_line_scratch(gray: np.ndarray, theta_steps: int = 180, threshold: int = 50,
                       c_sigma: float = 1.0, c_lo: float = 0.05, c_hi: float = 0.15) -> dict:
    t0 = time.perf_counter()
    canny_out = canny_scratch(gray, sigma=c_sigma, t_lo=c_lo, t_hi=c_hi)
    edges = (canny_out["hysteresis"] > 0.5)
    
    H, W = edges.shape
    x_r, y_r = W // 2, H // 2
    theta_vals = np.linspace(0, np.pi, theta_steps, endpoint=False)
    r_max = int(np.hypot(W, H)) + 1
    accumulator = np.zeros((2 * r_max, theta_steps), dtype=np.int64)
    ys, xs = np.nonzero(edges)
    
    if len(xs) > 0:
        u = (xs - x_r)[:, np.newaxis].astype(np.float64)
        v = (ys - y_r)[:, np.newaxis].astype(np.float64)
        cos_t = np.cos(theta_vals)[np.newaxis, :]
        sin_t = np.sin(theta_vals)[np.newaxis, :]
        r_mat = u * cos_t + v * sin_t
        j_mat = np.round(r_mat).astype(int) + r_max
        valid = (j_mat >= 0) & (j_mat < 2 * r_max)
        k_mat = np.tile(np.arange(theta_steps), (len(xs), 1))
        np.add.at(accumulator, (j_mat[valid], k_mat[valid]), 1)
        
    line_pos = np.argwhere(accumulator > threshold)
    result_rgb = _gray_to_rgb_uint8(gray)
    for j, k in line_pos[:60]:
        _draw_line_parametric(result_rgb, theta_vals[k], float(j - r_max), H, W, color=(0, 220, 0))
        
    acc_disp = np.log1p(accumulator.astype(np.float64))
    acc_disp = acc_disp / acc_disp.max() if acc_disp.max() > 0 else acc_disp
    
    return dict(edges=edges.astype(np.float64), accumulator=acc_disp, result_rgb=result_rgb, n_lines=len(line_pos), elapsed=(time.perf_counter()-t0)*1000)

def hough_line_library(gray: np.ndarray, c_sigma: float = 1.0, c_lo: float = 0.05, c_hi: float = 0.15) -> dict:
    t0 = time.perf_counter()
    edges = sk_canny(gray, sigma=c_sigma, low_threshold=c_lo, high_threshold=c_hi, use_quantiles=True)
    H, W = edges.shape
    h_space, theta, d = sk_hough_line(edges)
    
    acc_disp = np.log1p(h_space.astype(np.float64).T)
    acc_disp = acc_disp / acc_disp.max() if acc_disp.max() > 0 else acc_disp
    
    _, angles_p, dists_p = sk_hough_line_peaks(h_space, theta, d, num_peaks=20, min_distance=10, min_angle=10)
    result_rgb = _gray_to_rgb_uint8(gray)
    for angle, dist in zip(angles_p, dists_p):
        _draw_line_parametric(result_rgb, angle, dist, H, W, color=(0, 220, 0))
        
    return dict(edges=edges.astype(np.float64), accumulator=acc_disp, result_rgb=result_rgb, n_lines=len(angles_p), elapsed=(time.perf_counter()-t0)*1000)

def hough_circle_scratch(gray: np.ndarray, radius: int = 15, threshold_frac: float = 0.45,
                         c_sigma: float = 1.0, c_lo: float = 0.05, c_hi: float = 0.15) -> dict:
    t0 = time.perf_counter()
    canny_out = canny_scratch(gray, sigma=c_sigma, t_lo=c_lo, t_hi=c_hi)
    edges = (canny_out["hysteresis"] > 0.5)
    
    H, W = edges.shape
    accumulator = np.zeros((H, W), dtype=np.float64)
    ys, xs = np.nonzero(edges)
    
    if len(xs) > 0:
        t_deg = np.arange(0, 360, 2)
        cos_t, sin_t = np.cos(np.deg2rad(t_deg)), np.sin(np.deg2rad(t_deg))
        xs_col, ys_col = xs[:, np.newaxis].astype(np.float64), ys[:, np.newaxis].astype(np.float64)
        a_all  = (xs_col - radius * cos_t[np.newaxis, :]).astype(int)
        b_all  = (ys_col - radius * sin_t[np.newaxis, :]).astype(int)
        valid  = (a_all >= 0) & (a_all < W) & (b_all >= 0) & (b_all < H)
        np.add.at(accumulator, (b_all[valid], a_all[valid]), 1)
        
    acc_max = accumulator.max()
    centers = np.argwhere(accumulator > (threshold_frac * acc_max if acc_max > 0 else 1.0))
    
    result_rgb = _gray_to_rgb_uint8(gray)
    for cy, cx in centers[:40]:
        _draw_circle_on_rgb(result_rgb, cx, cy, radius, H, W, color=(0, 220, 0))
        
    acc_disp = accumulator / acc_max if acc_max > 0 else accumulator
    
    return dict(edges=edges.astype(np.float64), accumulator=acc_disp, result_rgb=result_rgb, n_circles=len(centers), elapsed=(time.perf_counter()-t0)*1000)

def hough_circle_library(gray: np.ndarray, c_sigma: float = 1.0, c_lo: float = 0.05, c_hi: float = 0.15) -> dict:
    t0 = time.perf_counter()
    edges = sk_canny(gray, sigma=c_sigma, low_threshold=c_lo, high_threshold=c_hi, use_quantiles=True)
    H, W = edges.shape
    radii = np.arange(8, 36, 3)
    
    h_circles = sk_hough_circle(edges, radii)
    accums, cx_arr, cy_arr, radii_det = sk_hough_circle_peaks(h_circles, radii, num_peaks=15, threshold=0.3)
    acc_disp = h_circles.max(axis=0) if len(h_circles) > 0 else np.zeros((H, W))
    
    result_rgb = _gray_to_rgb_uint8(gray)
    for cx, cy, r in zip(cx_arr, cy_arr, radii_det):
        _draw_circle_on_rgb(result_rgb, int(cx), int(cy), float(r), H, W, color=(0, 220, 0))
        
    return dict(edges=edges.astype(np.float64), accumulator=acc_disp.astype(np.float64), result_rgb=result_rgb, n_circles=len(cx_arr), elapsed=(time.perf_counter()-t0)*1000)