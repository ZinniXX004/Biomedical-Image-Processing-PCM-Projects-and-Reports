import time
import numpy as np
from skimage.feature import canny as sk_canny
from skimage.morphology import closing, disk, skeletonize
from skimage.measure import label as sk_label

from config import KERNELS, KIRSCH_K
from core.math_ops import manual_gaussian_filter, manual_convolve2d, normalize

def compute_traditional_pipeline(acq, method, sigma, thr):
    """Computes traditional edge detection (Prewitt, Sobel, Kirsch, etc.)"""
    t0 = time.perf_counter()
    denoised = manual_gaussian_filter(acq, sigma=sigma)
    morpho_pre = closing(denoised, disk(1))

    img = morpho_pre
    if method == "Kirsch":
        responses = [manual_convolve2d(img, k) for k in KIRSCH_K.values()]
        mag = np.max(np.stack(responses, axis=0), axis=0)
        gx = manual_convolve2d(img, KIRSCH_K["E"])
        gy = manual_convolve2d(img, KIRSCH_K["S"])
    else:
        gx = manual_convolve2d(img, KERNELS[method]["Gx"])
        gy = manual_convolve2d(img, KERNELS[method]["Gy"])
        mag = np.hypot(gx, gy)

    nm = normalize(mag)
    ang_nm = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    binary = (nm > thr).astype(np.float64)

    try:
        skel = skeletonize(binary > 0.5).astype(np.float64)
    except Exception:
        skel = binary.copy()

    elapsed = (time.perf_counter() - t0) * 1000

    return dict(
        denoised=denoised, morpho_pre=morpho_pre,
        gx=normalize(gx), gy=normalize(gy), magnitude=nm, direction=ang_nm,
        binary=binary, morpho_post=skel, elapsed=elapsed,
        density=float(binary.mean()), mean_mag=float(nm.mean()),
    )

# --- CANNY HELPER FUNCTIONS ---
def _digitize_angle(angle_deg: np.ndarray) -> np.ndarray:
    q = np.zeros_like(angle_deg, dtype=np.int32)
    a = angle_deg
    q[((a >= 0)    & (a <= 22.5))   | ((a > 157.5) & (a <= 202.5)) | ((a > 337.5) & (a <= 360))]   = 0
    q[((a > 22.5)  & (a <= 67.5))   | ((a > 202.5) & (a <= 247.5))] = 1
    q[((a > 67.5)  & (a <= 112.5))  | ((a > 247.5) & (a <= 292.5))] = 2
    q[((a > 112.5) & (a <= 157.5))  | ((a > 292.5) & (a <= 337.5))] = 3
    return q

def _non_max_suppression(quantized: np.ndarray, mag: np.ndarray) -> np.ndarray:
    p = np.pad(mag, 1, mode='edge')
    n_e, n_w   = p[1:-1, 2:], p[1:-1, :-2]
    n_n, n_s   = p[:-2, 1:-1], p[2:, 1:-1]
    n_ne, n_sw = p[:-2, 2:], p[2:, :-2]
    n_nw, n_se = p[:-2, :-2], p[2:, 2:]

    is_max = (
        ((quantized == 0) & (mag >= n_w)  & (mag >= n_e))   | 
        ((quantized == 1) & (mag >= n_sw) & (mag >= n_ne))  | 
        ((quantized == 2) & (mag >= n_n)  & (mag >= n_s))   | 
        ((quantized == 3) & (mag >= n_nw) & (mag >= n_se))
    )
    return np.where(is_max, mag, 0.0)

def _double_threshold(nms: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    result = np.zeros_like(nms)
    result[nms >= t_hi] = 255.0
    result[(nms >= t_lo) & (nms < t_hi)] = 128.0
    return result

def _hysteresis(double_thresh: np.ndarray) -> np.ndarray:
    strong = double_thresh >= 200
    weak   = double_thresh >= 100
    labeled, _ = sk_label(weak, return_num=True, connectivity=2)
    strong_ids = set(int(x) for x in np.unique(labeled[strong]) if x != 0)
    result = np.zeros_like(double_thresh)
    for sid in strong_ids:
        result[labeled == sid] = 255.0
    return result

def canny_scratch(gray: np.ndarray, sigma: float = 1.0, t_lo: float = 0.05, t_hi: float = 0.15) -> dict:
    t0 = time.perf_counter()
    smoothed = manual_gaussian_filter(gray, sigma)
    
    k = max(1, int(np.ceil(3.0 * sigma)))
    ax = np.arange(-k, k + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    gauss = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    gx_kernel = -(xx / sigma**2) * gauss
    gy_kernel = -(yy / sigma**2) * gauss

    fx = manual_convolve2d(smoothed, gx_kernel)
    fy = manual_convolve2d(smoothed, gy_kernel)
    mag = np.hypot(fx, fy)
    mag_max = mag.max() + 1e-9
    
    angle_deg = np.rad2deg(np.arctan2(fy, fx)) + 180.0
    quantized = _digitize_angle(angle_deg)
    
    color_rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    edge_mask = mag > (0.01 * mag_max)
    color_rgb[edge_mask & (quantized == 0), 0] = 255
    color_rgb[edge_mask & (quantized == 1), 1] = 255
    color_rgb[edge_mask & (quantized == 2), 2] = 255
    color_rgb[edge_mask & (quantized == 3), 0] = 255
    color_rgb[edge_mask & (quantized == 3), 1] = 255

    nms = _non_max_suppression(quantized, mag)
    double_thresh = _double_threshold(nms, t_lo * mag_max, t_hi * mag_max)
    final_edges = _hysteresis(double_thresh)

    elapsed = (time.perf_counter() - t0) * 1000
    return dict(
        smoothed=smoothed, fx=normalize(fx), fy=normalize(fy),
        magnitude=mag / mag_max, angle_disp=angle_deg / 360.0,
        quantized_disp=quantized.astype(np.float64) / 3.0,
        color_rgb=color_rgb, nms=nms / mag_max,
        double_thresh_disp=double_thresh / 255.0,
        hysteresis=final_edges / 255.0, elapsed=elapsed,
        density=float((final_edges > 127).mean()),
    )

def canny_library(gray: np.ndarray, sigma: float = 1.0, t_lo: float = 0.05, t_hi: float = 0.15) -> dict:
    t0 = time.perf_counter()
    result = sk_canny(gray, sigma=sigma, low_threshold=t_lo, high_threshold=t_hi, use_quantiles=True)
    return dict(result=result.astype(np.float64), elapsed=(time.perf_counter() - t0) * 1000, density=float(result.mean()))