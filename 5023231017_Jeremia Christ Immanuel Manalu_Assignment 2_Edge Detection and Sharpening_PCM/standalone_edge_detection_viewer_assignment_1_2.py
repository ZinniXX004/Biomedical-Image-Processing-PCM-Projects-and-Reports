#!/usr/bin/env python3
"""
Interactive Edge Detection Viewer
Multi-tab pipeline with per-subpanel interactive parameter controls.

MANUAL MATH IMPLEMENTATION: 
Replaced scipy's convolution and gaussian_filter, as well as skimage's rgb2gray with explicit mathematical algorithms using pure NumPy.

Run:
    python edge_detection_viewer_v3.py

Requirements:
    pip install PyQt6 matplotlib scikit-image numpy opencv-python pillow
"""

import sys, os, time, datetime
import numpy as np
from skimage import data as skdata, exposure
from skimage.morphology import closing, disk, skeletonize
from skimage.metrics import (
    structural_similarity  as _ssim_func,
    peak_signal_noise_ratio as _psnr_func,
    mean_squared_error      as _mse_func,
)
from skimage.measure import shannon_entropy as _shannon_entropy, label as sk_label
from skimage.feature import canny as sk_canny
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use("QtAgg")

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QSlider, QComboBox, QPushButton,
        QStatusBar, QFileDialog, QScrollArea, QFrame, QSizePolicy,
        QTabWidget,
    )
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPixmap, QImage
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavToolbar,
    )
    from matplotlib.figure import Figure
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install:  pip install PyQt6 matplotlib scikit-image numpy")
    sys.exit(1)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Steam-inspired Colour Theme
BG_DARK   = "#1b2838" 
BG_PANEL  = "#2a475e" 
BG_CARD   = "#171a21" 
TEXT_COL  = "#c7d5e0" 
SUBTEXT   = "#8f98a0" 
GRID_COL  = "#415467" 

STAGE_COLORS = {
    "preprocessing": "#66c0f4",
    "gradient":      "#a4d007",
    "results":       "#e3a968",
    "enhancement":   "#b4a6fb",
    "restoration":   "#4f94bc",
    "canny":         "#FF8C8C",   # soft red  — Canny tab
    "sharpening":    "#72E8A0",   # mint      — Sharpening tab
    "comparison":    "#FFD27B",   # amber     — Runtime and Comparison tab
}

# 4-direction colormap for Canny angle quantization
CANNY_DIR_CMAP = LinearSegmentedColormap.from_list(
    "canny_dir", ["#FF4444", "#44FF44", "#4444FF", "#FFFF44"], N=4)

METHOD_COLORS = {
    "Prewitt":        "#66c0f4",
    "Sobel":          "#a4d007",
    "Roberts":        "#F38BA8",
    "Extended Sobel": "#b4a6fb",
    "Kirsch":         "#e3a968",
}

ENH_COLORS = {
    "CS":    "#66c0f4",
    "HE":    "#a4d007",
    "CLAHE": "#b4a6fb",
}

EDGE_CMAP = LinearSegmentedColormap.from_list(
    "edge_glow",["#000000", "#171a21", "#1b2838", "#66c0f4", "#ffffff"])

# Kernels 
KERNELS = {
    "Prewitt": {
        "Gx": np.array([[-1, 0, 1], [-1, 0, 1],[-1, 0, 1]], dtype=np.float64),
        "Gy": np.array([[-1,-1,-1],[ 0, 0, 0], [ 1, 1, 1]], dtype=np.float64),
    },
    "Sobel": {
        "Gx": np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]], dtype=np.float64),
        "Gy": np.array([[-1,-2,-1],[ 0, 0, 0], [ 1, 2, 1]], dtype=np.float64),
    },
    "Roberts": {
        "Gx": np.array([[ 1,  0], [ 0, -1]], dtype=np.float64),
        "Gy": np.array([[ 0,  1], [-1,  0]], dtype=np.float64),
    },
    "Extended Sobel": {
        "Gx": np.array([[-1, -2,  0,  2,  1],[-4, -8,  0,  8,  4],
            [-6,-12,  0, 12,  6],[-4, -8,  0,  8,  4],[-1, -2,  0,  2,  1]], dtype=np.float64),
        "Gy": np.array([[-1, -4, -6, -4, -1],[-2, -8,-12, -8, -2],[ 0,  0,  0,  0,  0],[ 2,  8, 12,  8,  2],[ 1,  4,  6,  4,  1]], dtype=np.float64),
    },
    "Kirsch": None,
}

KIRSCH_K = {
    "N":  np.array([[ 5,  5,  5],[-3,  0, -3], [-3, -3, -3]], dtype=np.float64),
    "NE": np.array([[-3,  5,  5],[-3,  0,  5],[-3, -3, -3]], dtype=np.float64),
    "E":  np.array([[-3, -3,  5],[-3,  0,  5],[-3, -3,  5]], dtype=np.float64),
    "SE": np.array([[-3, -3, -3],[-3,  0,  5],[-3,  5,  5]], dtype=np.float64),
    "S":  np.array([[-3, -3, -3], [-3,  0, -3],[ 5,  5,  5]], dtype=np.float64),
    "SW": np.array([[-3, -3, -3],[ 5,  0, -3],[ 5,  5, -3]], dtype=np.float64),
    "W":  np.array([[ 5, -3, -3],[ 5,  0, -3],[ 5, -3, -3]], dtype=np.float64),
    "NW": np.array([[ 5,  5, -3],[ 5,  0, -3], [-3, -3, -3]], dtype=np.float64),
}

KERNEL_INFO = {
    "Prewitt":        "Size: 3×3 | Kernels: 2 (Gx, Gy)\nWeights: uniform ±1 | Noise: medium",
    "Sobel":          "Size: 3×3 | Kernels: 2 (Gx, Gy)\nWeights: ±1,±2 center | Noise: med-high",
    "Roberts":        "Size: 2×2 | Kernels: 2 (diagonal)\nWeights: ±1 diagonal | Noise: low (fast)",
    "Extended Sobel": "Size: 5×5 | Kernels: 2 (Gx, Gy)\nWeights: ±1..±12 | Noise: high",
    "Kirsch":         "Size: 3×3 | Kernels: 8 (compass dirs)\nDirections: N,NE…NW | Noise: high",
}


# MANUAL MATHEMATICAL IMPLEMENTATIONS
def manual_rgb2gray(rgb_img):
    """
    Manual RGB to Grayscale conversion using the Luminosity Method.
    Formula: Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
    """
    if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 3:
        r = rgb_img[..., 0]
        g = rgb_img[..., 1]
        b = rgb_img[..., 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        
        # Normalize to[0.0, 1.0] to match standard image processing scales
        if gray.max() > 1.0:
            gray = gray / 255.0
        return gray
    return rgb_img

def manual_convolve2d(image, kernel):
    """
    Manual 2D Convolution (Cross-correlation) using NumPy array slicing.
    Avoids Python nested loops per pixel to ensure UI remains highly responsive.
    """
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    # Pad image using edge reflection to mathematically handle borders
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(image, dtype=np.float64)
    
    # Slide the kernel over the image
    for i in range(k_h):
        for j in range(k_w):
            out += padded[i:i+image.shape[0], j:j+image.shape[1]] * kernel[i, j]
            
    return out

def manual_gaussian_filter(image, sigma):
    """
    Manual Gaussian Blur utilizing Mathematical Separability.
    Instead of an N x N convolution (which is very slow), we apply 
    a 1D horizontal pass followed by a 1D vertical pass.
    Formula: G(x) = exp(-(x^2) / (2 * sigma^2))
    """
    if sigma <= 1e-6:
        return image.copy()
        
    # Determine appropriate kernel size based on sigma distribution
    size = int(6 * sigma)
    if size % 2 == 0: size += 1
    if size < 3: size = 3
    
    k = size // 2
    x = np.arange(-k, k + 1)
    
    # Generate 1D Gaussian kernel
    g1d = np.exp(-(x**2) / (2 * sigma**2))
    g1d = g1d / g1d.sum()  # Normalize sum to 1.0
    
    # Reshape for Separable Convolution (1xN and Nx1)
    kernel_x = g1d.reshape(1, -1)
    kernel_y = g1d.reshape(-1, 1)
    
    # Pass 1: Horizontal, Pass 2: Vertical
    img_x = manual_convolve2d(image, kernel_x)
    return manual_convolve2d(img_x, kernel_y)

def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


# IMAGE PROCESSING PIPELINE
def compute_pipeline(gray, method, sigma, thr, enh_method="CLAHE", clahe_clip=0.03, cs_low=2.0, cs_high=98.0):
    t0 = time.perf_counter()

    acq = np.clip(gray.copy(), 0.0, 1.0)

    # 1. Enhancement
    if enh_method == "CLAHE":
        enhanced = exposure.equalize_adapthist(acq, clip_limit=clahe_clip)
    elif enh_method == "HE":
        enhanced = exposure.equalize_hist(acq)
    elif enh_method == "CS":
        p2  = np.percentile(acq, cs_low)
        p98 = np.percentile(acq, cs_high)
        enhanced = np.clip((acq - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)
    else: 
        enhanced = acq.copy()

    # 2. Restoration / Denoising (Using manual mathematical Gaussian)
    denoised = manual_gaussian_filter(enhanced, sigma=sigma) 
    morpho_pre = closing(denoised, disk(1))

    # 3. Gradient Computation (Using our manual 2D convolution)
    img = morpho_pre
    if method == "Kirsch":
        responses =[manual_convolve2d(img, k) for k in KIRSCH_K.values()]
        mag = np.max(np.stack(responses, axis=0), axis=0)
        gx  = manual_convolve2d(img, KIRSCH_K["E"])
        gy  = manual_convolve2d(img, KIRSCH_K["S"])
    else:
        kx  = KERNELS[method]["Gx"]
        ky  = KERNELS[method]["Gy"]
        gx  = manual_convolve2d(img, kx)
        gy  = manual_convolve2d(img, ky)
        mag = np.hypot(gx, gy)

    # 4. Normalization and Edge Mathematics
    nm = normalize(mag)
    ang = np.arctan2(gy, gx)
    ang_nm = (ang + np.pi) / (2 * np.pi)
    binary = (nm > thr).astype(np.float64)

    try:
        skel = skeletonize(binary > 0.5).astype(np.float64)
    except Exception:
        skel = binary.copy()

    elapsed = (time.perf_counter() - t0) * 1000

    return dict(
        acq=acq, enhanced=enhanced, denoised=denoised, morpho_pre=morpho_pre,
        gx=normalize(gx), gy=normalize(gy), magnitude=nm, direction=ang_nm,
        binary=binary, morpho_post=skel, elapsed=elapsed,
        density=float(binary.mean()), mean_mag=float(nm.mean()),
    )


def contrast_stretching(img, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    p2  = np.percentile(img, p_low)
    p98 = np.percentile(img, p_high)
    return np.clip((img - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)


def compute_all_enhancements(acq: np.ndarray, clahe_clip: float = 0.03, cs_low: float = 2.0, cs_high: float = 98.0) -> dict:
    return {
        "CS":    contrast_stretching(acq, cs_low, cs_high),
        "HE":    exposure.equalize_hist(acq),
        "CLAHE": exposure.equalize_adapthist(acq, clip_limit=clahe_clip),
    }


def compute_enhancement_metrics(reference: np.ndarray, enhanced: np.ndarray) -> dict:
    mse  = float(_mse_func(reference, enhanced))
    rmse = float(np.sqrt(mse))
    psnr = float(_psnr_func(reference, enhanced, data_range=1.0))
    if np.isinf(psnr) or np.isnan(psnr): psnr = 100.0
    ssim = float(_ssim_func(reference, enhanced, data_range=1.0))
    entropy = float(_shannon_entropy(enhanced))
    return {"RMSE": rmse, "PSNR": psnr, "SSIM": ssim, "Entropy": entropy}


# Canny Edge Detection (from scratch)

def _digitize_angle(angle_deg: np.ndarray) -> np.ndarray:
    """
    Vectorized angle digitization into 4 gradient directions.
    Following lecture Week 10 exactly:
      0 = Horizontal  (0°–22.5°, 157.5°–202.5°, 337.5°–360°)
      1 = Diagonal↗   (22.5°–67.5°, 202.5°–247.5°)
      2 = Vertical    (67.5°–112.5°, 247.5°–292.5°)
      3 = Diagonal↘   (112.5°–157.5°, 292.5°–337.5°)
    """
    q = np.zeros_like(angle_deg, dtype=np.int32)
    a = angle_deg
    q[((a >= 0)    & (a <= 22.5))   |
      ((a > 157.5) & (a <= 202.5))  |
      ((a > 337.5) & (a <= 360))]   = 0
    q[((a > 22.5)  & (a <= 67.5))   |
      ((a > 202.5) & (a <= 247.5))] = 1
    q[((a > 67.5)  & (a <= 112.5))  |
      ((a > 247.5) & (a <= 292.5))] = 2
    q[((a > 112.5) & (a <= 157.5))  |
      ((a > 292.5) & (a <= 337.5))] = 3
    return q


def _non_max_suppression(quantized: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """
    Vectorized Non-Maximum Suppression (NMS).
    Keeps pixels that are local maxima along their gradient direction.
    Implements the same logic as the lecture's Non_Max_Supp() but via
    NumPy slicing to avoid slow Python pixel loops.
    """
    p = np.pad(mag, 1, mode='edge')
    # Pre-fetch all 8 neighbour planes
    n_e  = p[1:-1, 2:]    # East
    n_w  = p[1:-1, :-2]   # West
    n_n  = p[:-2,  1:-1]  # North
    n_s  = p[2:,   1:-1]  # South
    n_ne = p[:-2,  2:]    # North-East
    n_sw = p[2:,   :-2]   # South-West
    n_nw = p[:-2,  :-2]   # North-West
    n_se = p[2:,   2:]    # South-East

    is_max = (
        ((quantized == 0) & (mag >= n_w)  & (mag >= n_e))   |   # horizontal
        ((quantized == 1) & (mag >= n_sw) & (mag >= n_ne))  |   # 45° diagonal
        ((quantized == 2) & (mag >= n_n)  & (mag >= n_s))   |   # vertical
        ((quantized == 3) & (mag >= n_nw) & (mag >= n_se))      # 135° diagonal
    )
    return np.where(is_max, mag, 0.0)


def _double_threshold(nms: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    """
    Double thresholding (vectorized):
      < t_lo  → 0   (no edge)
      t_lo–t_hi → 128 (weak edge)
      >= t_hi → 255 (strong edge)
    """
    result = np.zeros_like(nms)
    result[nms >= t_hi]                       = 255.0
    result[(nms >= t_lo) & (nms < t_hi)]      = 128.0
    return result


def _hysteresis(double_thresh: np.ndarray) -> np.ndarray:
    """
    Edge tracking by hysteresis using skimage connected-component labelling.
    Weak edges (128) are kept only when connected to at least one strong edge (255).
    """
    strong = double_thresh >= 200
    weak   = double_thresh >= 100          # both weak + strong pixels

    labeled, _ = sk_label(weak, return_num=True, connectivity=2)

    # Component IDs that contain a strong pixel are kept
    strong_ids = set(int(x) for x in np.unique(labeled[strong]) if x != 0)

    result = np.zeros_like(double_thresh)
    for sid in strong_ids:
        result[labeled == sid] = 255.0
    return result


def canny_scratch(gray: np.ndarray, sigma: float = 1.0,
                  t_lo: float = 0.05, t_hi: float = 0.15) -> dict:
    """
    Full Canny edge detector implemented from scratch.
    Follows lecture Week 10 step-by-step methodology.

    Parameters
    gray  : float64 grayscale image [0,1]
    sigma : Gaussian σ for smoothing
    t_lo  : low threshold  (fraction of max gradient magnitude)
    t_hi  : high threshold (fraction of max gradient magnitude)

    Returns dict with all intermediate stages for visualisation.
    """
    t0 = time.perf_counter()

    # Step 1: Gaussian smoothing  Ī = I * H^{G,σ}
    smoothed = manual_gaussian_filter(gray, sigma)

    # Step 2: Gaussian gradient kernels  (analytic derivatives)
    # Īx = -(x/σ²) · exp(-(x²+y²)/σ²)
    # Īy = -(y/σ²) · exp(-(x²+y²)/σ²)
    k  = max(1, int(np.ceil(3.0 * sigma)))
    ax = np.arange(-k, k + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    gauss      = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    gx_kernel  = -(xx / sigma**2) * gauss
    gy_kernel  = -(yy / sigma**2) * gauss

    fx = manual_convolve2d(smoothed, gx_kernel)
    fy = manual_convolve2d(smoothed, gy_kernel)

    # Step 3: Gradient magnitude  Emag = √(fx² + fy²)
    mag     = np.hypot(fx, fy)
    mag_max = mag.max() + 1e-9
    mag_norm = mag / mag_max

    # Step 4: Gradient direction  Φ(u,v) = arctan2(Iy, Ix) + 180
    angle_deg = np.rad2deg(np.arctan2(fy, fx)) + 180.0   # → [0°, 360°)

    # Step 5: Digitize angle → 4 directions
    quantized = _digitize_angle(angle_deg)

    # Color visualization: one RGB channel per direction (like the lecture slide)
    color_rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    edge_mask = mag > (0.01 * mag_max)
    color_rgb[edge_mask & (quantized == 0), 0] = 255                      # red
    color_rgb[edge_mask & (quantized == 1), 1] = 255                      # green
    color_rgb[edge_mask & (quantized == 2), 2] = 255                      # blue
    color_rgb[edge_mask & (quantized == 3), 0] = 255                      # yellow
    color_rgb[edge_mask & (quantized == 3), 1] = 255

    # Step 6: Non-Maximum Suppression 
    nms      = _non_max_suppression(quantized, mag)
    nms_norm = nms / mag_max

    #Step 7: Double Thresholding 
    t_hi_abs = t_hi * mag_max
    t_lo_abs = t_lo * mag_max
    double_thresh = _double_threshold(nms, t_lo_abs, t_hi_abs)

    # Step 8: Hysteresis Edge Tracking
    final_edges = _hysteresis(double_thresh)

    elapsed = (time.perf_counter() - t0) * 1000

    return dict(
        smoothed          = smoothed,
        fx                = normalize(fx),
        fy                = normalize(fy),
        magnitude         = mag_norm,
        angle_disp        = angle_deg / 360.0,        # [0,1] for HSV display
        quantized_disp    = quantized.astype(np.float64) / 3.0,  # 4 levels → [0,1]
        color_rgb         = color_rgb,                 # uint8 RGB, no cmap needed
        nms               = nms_norm,
        double_thresh_disp= double_thresh / 255.0,
        hysteresis        = final_edges  / 255.0,
        elapsed           = elapsed,
        density           = float((final_edges > 127).mean()),
    )


def canny_library(gray: np.ndarray, sigma: float = 1.0,
                  t_lo: float = 0.05, t_hi: float = 0.15) -> dict:
    """
    Canny edge detection using skimage.feature.canny.
    t_lo / t_hi are treated as quantile fractions (use_quantiles=True).
    """
    t0 = time.perf_counter()
    result = sk_canny(gray, sigma=sigma,
                      low_threshold=t_lo, high_threshold=t_hi,
                      use_quantiles=True)
    elapsed = (time.perf_counter() - t0) * 1000
    return dict(
        result  = result.astype(np.float64),
        elapsed = elapsed,
        density = float(result.mean()),
    )


# Image Sharpening

def laplacian_sharpening(gray: np.ndarray,
                          weight: float = 1.0,
                          kernel_type: str = "H4") -> dict:
    """
    Laplacian-based edge sharpening.   I' = I − w·(H^L * I)

    Kernel variants (from lecture):
      H4  = [[0,1,0],[1,-4,1],[0,1,0]]       (4-connectivity)
      H8  = [[1,1,1],[1,-8,1],[1,1,1]]       (8-connectivity)
      H12 = [[1,2,1],[2,-12,2],[1,2,1]]      (weighted 8-connectivity)

    Also computes the separable form using
      H_x = [1,-2,1]  and  H_y = [[1],[-2],[1]]
    """
    t0 = time.perf_counter()

    # Light pre-smoothing to reduce noise before Laplacian
    blur = manual_gaussian_filter(gray, sigma=1.0)

    # Separable Laplacian kernels (1-D)
    HL_x = np.array([[1.0, -2.0, 1.0]])
    HL_y = np.array([[1.0], [-2.0], [1.0]])

    lap_x      = manual_convolve2d(blur, HL_x)
    lap_y      = manual_convolve2d(blur, HL_y)
    lap_xy_sep = lap_x + lap_y
    sharp_sep  = np.clip(gray - weight * lap_xy_sep, 0.0, 1.0)

    # Full 2-D Laplacian kernel
    if kernel_type == "H8":
        HL = np.array([[1.,  1., 1.],
                       [1., -8., 1.],
                       [1.,  1., 1.]])
    elif kernel_type == "H12":
        HL = np.array([[1.,  2., 1.],
                       [2.,-12., 2.],
                       [1.,  2., 1.]])
    else:   # H4 (default, matches lecture HL = Hx + Hy)
        HL = np.array([[0.,  1., 0.],
                       [1., -4., 1.],
                       [0.,  1., 0.]])

    lap_full   = manual_convolve2d(blur, HL)
    sharp_full = np.clip(gray - weight * lap_full, 0.0, 1.0)

    elapsed = (time.perf_counter() - t0) * 1000

    return dict(
        blur       = blur,
        lap_x      = normalize(np.abs(lap_x)),
        lap_y      = normalize(np.abs(lap_y)),
        lap_xy_sep = normalize(np.abs(lap_xy_sep)),
        sharp_sep  = sharp_sep,
        lap_full   = normalize(np.abs(lap_full)),
        sharp_full = sharp_full,
        elapsed    = elapsed,
    )


def unsharp_masking(gray: np.ndarray, a: float = 0.7, sigma: float = 1.0) -> dict:
    """
    Unsharp Masking (USM) — from lecture equations:
      M  ← I − (I * H̃)    (mask = original − blurred)
      Ǐ  ← I + a · M      (sharpened = original + factor × mask)
    """
    t0 = time.perf_counter()
    blurred   = manual_gaussian_filter(gray, sigma=sigma)
    mask      = gray - blurred
    sharpened = np.clip(gray + a * mask, 0.0, 1.0)
    elapsed   = (time.perf_counter() - t0) * 1000
    return dict(
        blurred   = blurred,
        mask      = normalize(mask),     # normalized for display
        sharpened = sharpened,
        elapsed   = elapsed,
    )


def compute_all_methods(gray, sigma, thr, canny_sigma, canny_tlo, canny_thi,
                         lap_weight, lap_kernel, usm_a, usm_sigma,
                         enh_method, clahe_clip, cs_low, cs_high) -> dict:
    """
    Run ALL edge detection + sharpening methods and return
    {method_name: {"edge": ndarray, "elapsed": float}}.
    Used by the Runtime and Comparison tab.
    """
    results = {}
    for m in ["Prewitt", "Sobel", "Roberts", "Extended Sobel", "Kirsch"]:
        r = compute_pipeline(gray, m, sigma, thr, enh_method, clahe_clip, cs_low, cs_high)
        results[m] = {"edge": r["morpho_post"], "elapsed": r["elapsed"]}

    r = canny_scratch(gray, canny_sigma, canny_tlo, canny_thi)
    results["Canny (Scratch)"] = {"edge": r["hysteresis"], "elapsed": r["elapsed"]}

    r = canny_library(gray, canny_sigma, canny_tlo, canny_thi)
    results["Canny (Library)"] = {"edge": r["result"], "elapsed": r["elapsed"]}

    r = laplacian_sharpening(gray, lap_weight, lap_kernel)
    results["Laplacian"] = {"edge": r["sharp_full"], "elapsed": r["elapsed"]}

    r = unsharp_masking(gray, usm_a, usm_sigma)
    results["Unsharp Masking"] = {"edge": r["sharpened"], "elapsed": r["elapsed"]}

    return results


# Per-Stage Image Panel
class ImagePanel(QWidget):
    def __init__(self, title: str, subtitle: str, stage_category: str, parent=None):
        super().__init__(parent)
        self._colorbar = None
        self._border   = STAGE_COLORS.get(stage_category, GRID_COL)

        self.setObjectName("ImagePanel")
        self.setStyleSheet(
            f"#ImagePanel {{ border: 1px solid {GRID_COL}; border-radius: 8px; background-color: {BG_CARD}; }}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        hdr = QWidget()
        hdr.setObjectName("panelHdr")
        hdr.setStyleSheet(
            f"#panelHdr {{ background-color: {BG_PANEL}; border-radius: 6px; }}"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 8, 10, 8)
        hl.setSpacing(8)

        dot = QLabel("●")
        dot.setStyleSheet(f"color:{self._border}; font-size:10px; background:transparent; border:none;")

        t_lbl = QLabel(title)
        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        t_lbl.setStyleSheet(f"color:{TEXT_COL}; background:transparent; border:none;")

        s_lbl = QLabel(subtitle)
        s_lbl.setFont(QFont("Segoe UI", 9))
        s_lbl.setStyleSheet(f"color:{SUBTEXT}; background:transparent; border:none;")

        hl.addWidget(dot)
        hl.addWidget(t_lbl)
        hl.addStretch()
        hl.addWidget(s_lbl)
        root.addWidget(hdr)

        self.fig = Figure(figsize=(4.0, 3.2), facecolor=BG_CARD)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.ax  = self.fig.add_subplot(111)
        self._style_ax()

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color:{BG_CARD}; border:none; border-radius:6px;")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setMaximumHeight(30)
        self.toolbar.setStyleSheet(
            f"QToolBar {{ background:{BG_PANEL}; border:none; spacing:4px; padding: 2px; }}"
            f"QToolButton {{"
            f"  background:{BG_DARK}; border:1px solid {GRID_COL};"
            f"  border-radius:4px; color:{TEXT_COL}; padding:4px; min-width:24px;"
            f"}}"
            f"QToolButton:hover  {{ background:{GRID_COL}; border-color:{self._border}; }}"
            f"QToolButton:checked {{ background:{self._border}30; border-color:{self._border}; }}"
        )

        root.addWidget(self.canvas, 1)
        root.addWidget(self.toolbar)
        self.setMinimumSize(340, 300)

    def _style_ax(self):
        self.ax.set_facecolor(BG_PANEL)
        for sp in self.ax.spines.values():
            sp.set_color(GRID_COL)
        self.ax.tick_params(colors=SUBTEXT, labelsize=8)

    def _reset_ax(self):
        if self._colorbar is not None:
            try: self._colorbar.remove()
            except Exception: pass
            self._colorbar = None
        self.ax.clear()
        self._style_ax()

    def show_image(self, img, cmap="gray", vmin=0.0, vmax=1.0, colorbar=False):
        self._reset_ax()
        im = self.ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                            interpolation="bilinear", aspect="equal")
        self.ax.axis("off")
        if colorbar:
            self._colorbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.02)
            self._colorbar.ax.tick_params(colors=SUBTEXT, labelsize=8)
            for sp in self._colorbar.ax.spines.values():
                sp.set_color(GRID_COL)
        self.canvas.draw_idle()

    def show_histogram(self, nm, thr, method_color):
        self._reset_ax()
        hist, bins = np.histogram(nm.ravel(), bins=80, range=(0, 1))
        self.ax.fill_between(bins[:-1], 0, hist, alpha=0.35, color=method_color)
        self.ax.plot(bins[:-1], hist, color=method_color, linewidth=1.5)
        self.ax.axvline(thr, color="white", linestyle="--", linewidth=1.5, label=f"thr={thr:.2f}")
        self.ax.set_xlabel("Magnitude", fontsize=9, color=SUBTEXT)
        self.ax.set_ylabel("Count",     fontsize=9, color=SUBTEXT)
        self.ax.legend(fontsize=8, loc="upper right", facecolor=BG_PANEL, edgecolor=GRID_COL, labelcolor=TEXT_COL)
        self.ax.grid(True, alpha=0.15, color=SUBTEXT)
        self.fig.subplots_adjust(left=0.15, right=0.92, top=0.92, bottom=0.18)
        self.canvas.draw_idle()

    def show_profile(self, nm, binary, thr, method_color, row):
        self._reset_ax()
        self.ax.plot(nm[row, :],     color=method_color, lw=1.8, label="Magnitude")
        self.ax.plot(binary[row, :], color="white",      lw=1.0, alpha=0.5, label="Binary")
        self.ax.axhline(thr, color="yellow", lw=1.2, linestyle=":", label=f"thr={thr:.2f}")
        self.ax.set_xlabel("Column",    fontsize=9, color=SUBTEXT)
        self.ax.set_ylabel("Intensity", fontsize=9, color=SUBTEXT)
        self.ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COL, labelcolor=TEXT_COL)
        self.ax.grid(True, alpha=0.15, color=SUBTEXT)
        self.ax.set_ylim(-0.05, 1.1)
        self.fig.subplots_adjust(left=0.15, right=0.92, top=0.92, bottom=0.18)
        self.canvas.draw_idle()

    def save_to(self, filepath: str) -> bool:
        try:
            self.fig.savefig(filepath, dpi=120, bbox_inches="tight", facecolor=BG_CARD, edgecolor="none")
            return True
        except Exception as e:
            print(f"[SAVE ERROR] {filepath}: {e}")
            return False


# Analysis Panel
class AnalysisPanel(QWidget):
    def __init__(self, title: str, subtitle: str, stage_category: str, figsize: tuple = (4.0, 3.2), parent=None):
        super().__init__(parent)
        self._border = STAGE_COLORS.get(stage_category, GRID_COL)

        self.setObjectName("AnalysisPanel")
        self.setStyleSheet(
            f"#AnalysisPanel {{ border: 1px solid {GRID_COL}; border-radius: 8px; background-color: {BG_CARD}; }}"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        hdr = QWidget()
        hdr.setObjectName("apHdr")
        hdr.setStyleSheet(f"#apHdr {{ background-color: {BG_PANEL}; border-radius: 6px; }}")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 8, 10, 8)
        hl.setSpacing(8)

        dot = QLabel("◈")
        dot.setStyleSheet(f"color:{self._border}; font-size:12px; background:transparent; border:none;")
        t_lbl = QLabel(title)
        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        t_lbl.setStyleSheet(f"color:{TEXT_COL}; background:transparent; border:none;")
        s_lbl = QLabel(subtitle)
        s_lbl.setFont(QFont("Segoe UI", 9))
        s_lbl.setStyleSheet(f"color:{SUBTEXT}; background:transparent; border:none;")

        hl.addWidget(dot)
        hl.addWidget(t_lbl)
        hl.addStretch()
        hl.addWidget(s_lbl)
        root.addWidget(hdr)

        self.fig = Figure(figsize=figsize, facecolor=BG_CARD)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color:{BG_CARD}; border:none; border-radius:6px;")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setMaximumHeight(30)
        self.toolbar.setStyleSheet(
            f"QToolBar {{ background:{BG_PANEL}; border:none; spacing:4px; padding: 2px; }}"
            f"QToolButton {{"
            f"  background:{BG_DARK}; border:1px solid {GRID_COL};"
            f"  border-radius:4px; color:{TEXT_COL}; padding:4px; min-width:24px;"
            f"}}"
            f"QToolButton:hover  {{ background:{GRID_COL}; border-color:{self._border}; }}"
            f"QToolButton:checked {{ background:{self._border}30; border-color:{self._border}; }}"
        )

        root.addWidget(self.canvas, 1)
        root.addWidget(self.toolbar)
        self.setMinimumSize(340, 300)

    def _fresh_ax(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(BG_PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID_COL)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        return ax

    @staticmethod
    def _style_twin(ax2):
        for sp in ax2.spines.values():
            sp.set_color(GRID_COL)
        ax2.tick_params(colors=SUBTEXT, labelsize=8)

    def show_hist_ogive(self, img: np.ndarray, label: str = "", color: str = "#89B4FA") -> None:
        ax = self._fresh_ax()
        hist, bins = np.histogram(img.ravel(), bins=80, range=(0.0, 1.0))
        cdf     = np.cumsum(hist) / img.size * 100.0
        centers = (bins[:-1] + bins[1:]) / 2.0
        bw      = centers[1] - centers[0]

        ax.bar(centers, hist, width=bw, color=color, alpha=0.55)
        ax.plot(centers, hist, color=color, lw=1.5, label="Histogram")
        ax.set_xlabel("Pixel Intensity", fontsize=9, color=SUBTEXT)
        ax.set_ylabel("Pixel Count",     fontsize=9, color=SUBTEXT)
        ax.grid(True, alpha=0.15, color=SUBTEXT)

        ax2 = ax.twinx()
        self._style_twin(ax2)
        ax2.plot(centers, cdf, color="white", lw=2.0, linestyle="--", alpha=0.9, label="Ogive (CDF %)")
        ax2.set_ylabel("Cumulative %", fontsize=9, color=SUBTEXT)
        ax2.set_ylim(0.0, 105.0)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COL, labelcolor=TEXT_COL, loc="upper left")
        if label:
            ax.set_title(label, fontsize=10, color=TEXT_COL, pad=6)

        self.fig.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.18)
        self.canvas.draw_idle()

    def show_comparison_hist_ogive(self, images_dict: dict, title: str = "") -> None:
        _cycle =["#89B4FA", "#FAB387", "#A6E3A1", "#CBA6F7"]
        ax = self._fresh_ax()
        ax2 = ax.twinx()
        self._style_twin(ax2)

        for i, (lbl, img) in enumerate(images_dict.items()):
            c = _cycle[i % len(_cycle)]
            hist, bins = np.histogram(img.ravel(), bins=80, range=(0.0, 1.0))
            cdf     = np.cumsum(hist) / img.size * 100.0
            centers = (bins[:-1] + bins[1:]) / 2.0
            ax.plot(centers,  hist, color=c, lw=1.8, alpha=0.85, label=f"{lbl} Hist")
            ax2.plot(centers, cdf,  color=c, lw=1.8, linestyle="--", alpha=0.85, label=f"{lbl} CDF")

        ax.set_xlabel("Pixel Intensity", fontsize=9, color=SUBTEXT)
        ax.set_ylabel("Pixel Count",     fontsize=9, color=SUBTEXT)
        ax2.set_ylabel("Cumulative %",   fontsize=9, color=SUBTEXT)
        ax2.set_ylim(0.0, 105.0)
        ax.grid(True, alpha=0.15, color=SUBTEXT)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COL, labelcolor=TEXT_COL, loc="upper left", ncol=2)
        if title:
            ax.set_title(title, fontsize=10, color=TEXT_COL, pad=6)

        self.fig.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.18)
        self.canvas.draw_idle()

    def show_performance_eval(self, metrics_per_method: dict) -> None:
        self.fig.clear()

        metric_cfg =[
            ("RMSE",    "↓ lower = better",          "#F38BA8"),
            ("PSNR",    "↑ higher = better[dB]",   "#A6E3A1"),
            ("SSIM",    "↑ higher = better  [0–1]",  "#89B4FA"),
            ("Entropy", "↑ higher = info[bits]",    "#CBA6F7"),
        ]
        methods   = list(metrics_per_method.keys())
        bar_clrs  =[ENH_COLORS.get(m, "#89B4FA") for m in methods]

        for idx, (mname, hint, accent) in enumerate(metric_cfg):
            ax = self.fig.add_subplot(2, 2, idx + 1)
            ax.set_facecolor(BG_PANEL)
            for sp in ax.spines.values():
                sp.set_color(GRID_COL)
            ax.tick_params(colors=SUBTEXT, labelsize=8)

            vals =[metrics_per_method[m].get(mname, 0.0) for m in methods]
            bars = ax.bar(methods, vals, color=bar_clrs, alpha=0.85, width=0.5,
                          edgecolor=GRID_COL, linewidth=1.0)

            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() * 1.02,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=8, color=TEXT_COL, fontfamily="Consolas"
                )

            ax.set_title(f"{mname} ({hint})", fontsize=10, color=accent, pad=6)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, fontsize=9, color=TEXT_COL, fontweight="bold")
            ax.grid(True, alpha=0.15, color=SUBTEXT, axis="y")

            _mn = min(vals)
            _mx = max(vals)
            ax.set_ylim(max(0.0, _mn * 0.88), _mx * 1.18 if _mx > 0 else 1.0)

        self.fig.suptitle("Enhancement Performance Evaluation (reference = acquired image)", fontsize=11, color=TEXT_COL, y=0.98)
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, hspace=0.65, wspace=0.35)
        self.canvas.draw_idle()

    def show_runtime_bars(self, runtimes_dict: dict) -> None:
        """Horizontal bar chart comparing method runtimes (green/orange/red)."""
        ax = self._fresh_ax()
        methods = list(runtimes_dict.keys())
        times   = [runtimes_dict[m] for m in methods]
        t_max   = max(times) + 1e-9

        bar_colors = []
        for t in times:
            ratio = t / t_max
            if ratio < 0.25:   bar_colors.append("#A6E3A1")
            elif ratio < 0.60: bar_colors.append("#E3A968")
            else:              bar_colors.append("#F38BA8")

        bars = ax.barh(methods, times, color=bar_colors, alpha=0.85,
                       edgecolor=GRID_COL, linewidth=0.8, height=0.55)
        for bar, t in zip(bars, times):
            ax.text(bar.get_width() + t_max * 0.012,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{t:.1f} ms", va="center", ha="left",
                    fontsize=9, color=TEXT_COL)

        ax.set_xlabel("Runtime (ms)", fontsize=10, color=SUBTEXT)
        ax.set_title("Method Runtime Comparison  ·  (green) fast   (orange) medium  (red) slow",
                     fontsize=11, color=TEXT_COL, pad=8)
        ax.grid(True, alpha=0.15, color=SUBTEXT, axis="x")
        ax.set_xlim(0, t_max * 1.32)
        self.fig.subplots_adjust(left=0.22, right=0.88, top=0.90, bottom=0.12)
        self.canvas.draw_idle()

    def show_image_grid(self, images_dict: dict, title: str = "",
                        cmap: str = "gray", cols: int = 4) -> None:
        """Grid of labelled images — multi-method visual comparison."""
        self.fig.clear()
        n    = len(images_dict)
        rows = max(1, (n + cols - 1) // cols)

        for idx, (lbl, img) in enumerate(images_dict.items()):
            ax = self.fig.add_subplot(rows, cols, idx + 1)
            ax.set_facecolor(BG_PANEL)
            if img is not None:
                if img.ndim == 3:
                    ax.imshow(img.astype(np.uint8), aspect="equal",
                              interpolation="bilinear")
                else:
                    ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0,
                              aspect="equal", interpolation="bilinear")
            ax.set_title(lbl, fontsize=8, color=TEXT_COL, pad=3)
            ax.axis("off")

        if title:
            self.fig.suptitle(title, fontsize=11, color=TEXT_COL, y=0.99)
        self.fig.subplots_adjust(
            left=0.02, right=0.98, top=0.93, bottom=0.02,
            hspace=0.38, wspace=0.06)
        self.canvas.draw_idle()

    def show_rgb_image(self, rgb_img: np.ndarray, title: str = "") -> None:
        """Display a uint8 RGB image (H,W,3) directly — no colormap needed."""
        ax = self._fresh_ax()
        ax.imshow(rgb_img.astype(np.uint8), aspect="equal",
                  interpolation="bilinear")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=10, color=TEXT_COL, pad=6)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
        self.canvas.draw_idle()

    def save_to(self, filepath: str) -> bool:
        try:
            self.fig.savefig(filepath, dpi=120, bbox_inches="tight", facecolor=BG_CARD, edgecolor="none")
            return True
        except Exception as e:
            print(f"[SAVE ERROR] {filepath}: {e}")
            return False

# Custom Background Widget (High Definition rendering)
class BackgroundWidget(QWidget):
    def __init__(self, image_path: str, opacity: float = 0.3, dimness: float = 0.6, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.opacity = opacity
        self.dimness = dimness
        
        # Load using QImage for the highest quality initial pixel data parsing
        self._original_image = QImage(self.image_path)
        self._cached_pixmap = None
        
    def resizeEvent(self, event):
        # Update the scaled image only when the window is resized for best performance and quality 
        super().resizeEvent(event)
        self._update_background_cache()

    def _update_background_cache(self):
        if self._original_image.isNull() or self.width() == 0 or self.height() == 0:
            return
            
        scale_mode = Qt.AspectRatioMode.KeepAspectRatioByExpanding
        scaled_img = self._original_image.scaled(
            self.size(), 
            scale_mode, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Convert to QPixmap purely for hardware-accelerated drawing
        self._cached_pixmap = QPixmap.fromImage(scaled_img)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Fill the base Steam background color
        painter.fillRect(self.rect(), QColor(BG_DARK))
        
        # Draw the high-quality cached pixmap
        if self._cached_pixmap and not self._cached_pixmap.isNull():
            painter.setOpacity(self.opacity)
            
            # Center the image perfectly
            x = (self.width() - self._cached_pixmap.width()) // 2
            y = (self.height() - self._cached_pixmap.height()) // 2
            painter.drawPixmap(x, y, self._cached_pixmap)
        
        # Draw the dimming overlay
        if self.dimness > 0:
            painter.setOpacity(self.dimness)
            painter.fillRect(self.rect(), QColor(BG_DARK))

# Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬 Interactive Edge Detection Viewer (Modern UI)")
        self.resize(1750, 1000)
        self._apply_dark_palette()

        self.gray_img     = None
        self.image_name   = "unknown"
        self._last_result = None

        self.enh_clahe_clip = 0.03
        self.cs_low         = 2.0
        self.cs_high        = 98.0

        # Canny + Sharpening state
        self.canny_sigma  = 1.0
        self.canny_t_lo   = 0.05
        self.canny_t_hi   = 0.15
        self.lap_weight   = 1.0
        self.lap_kernel   = "H4"
        self.usm_a        = 0.7
        self.usm_sigma    = 1.0
        # cached results (None = needs computation)
        self._last_canny_scratch  = None
        self._last_canny_lib      = None
        self._last_sharpening_lap = None
        self._last_sharpening_usm = None
        self._last_runtimes       = None
        self._runtime_dirty       = True

        self._load_default_image()

        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._do_update)

        self._autosave_timer = QTimer()
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._do_autosave)

        # Debounce timers for the two new tabs (longer delay = less UI lag)
        self._canny_debounce = QTimer()
        self._canny_debounce.setSingleShot(True)
        self._canny_debounce.timeout.connect(self._do_update_canny)

        self._sharp_debounce = QTimer()
        self._sharp_debounce.setSingleShot(True)
        self._sharp_debounce.timeout.connect(self._do_update_sharpening)

        self.panels          = {}
        self.analysis_panels = {}

        self._build_ui()
        self._update()
        # Trigger initial Canny + Sharpening after the pipeline settles
        self._canny_debounce.start(800)
        self._sharp_debounce.start(800)

    def _apply_dark_palette(self):
        p = QPalette()
        for role, hex_col in[
            (QPalette.ColorRole.Window,        BG_DARK),
            (QPalette.ColorRole.WindowText,    TEXT_COL),
            (QPalette.ColorRole.Base,          BG_PANEL),
            (QPalette.ColorRole.AlternateBase, BG_CARD),
            (QPalette.ColorRole.Text,          TEXT_COL),
            (QPalette.ColorRole.Button,        BG_CARD),
            (QPalette.ColorRole.ButtonText,    TEXT_COL),
        ]:
            p.setColor(role, QColor(hex_col))
        QApplication.instance().setPalette(p)

    def _load_default_image(self):
        astro = skdata.astronaut()
        h = min(astro.shape[0], 300)
        w = min(astro.shape[1], 300)
        self.gray_img   = manual_rgb2gray(astro[:h, :w])
        self.image_name = "Astronaut (skimage built-in)"

    def _build_ui(self):
        # ── CUSTOM BACKGROUND SETUP ──
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.custom_bg_path = os.path.join(base_dir, "Stultifera Navis.jpg")   
        
        self.bg_widget = BackgroundWidget(
            image_path=self.custom_bg_path,
            opacity=0.45,   # Image visibility (0.0 to 1.0)
            dimness=0.60    # Dark overlay intensity (0.0 to 1.0)
        )
        self.setCentralWidget(self.bg_widget)
        
        root = QHBoxLayout(self.bg_widget)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(16)

        self.left_panel = self._build_controls()
        root.addWidget(self.left_panel)
        
        self.tabs = self._build_tabbed_view()
        root.addWidget(self.tabs, 1)

        self.status = QStatusBar()
        self.status.setStyleSheet(
            f"QStatusBar{{ background: rgba(23, 26, 33, 0.90); color:{SUBTEXT}; font-size:11px; padding:6px; font-family:'Segoe UI'; border-top:1px solid {GRID_COL}; }}"
        )
        self.setStatusBar(self.status)
    
    # Custom UI Components
    def _create_card(self, title: str, accent_color: str) -> tuple[QWidget, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("ModernCard")
        card.setStyleSheet(f"""
            #ModernCard {{
                background-color: {BG_CARD};
                border: 1px solid {GRID_COL};
                border-radius: 10px;
            }}
        """)
        outer_layout = QVBoxLayout(card)
        outer_layout.setContentsMargins(14, 14, 14, 14)
        outer_layout.setSpacing(10)

        if title:
            header = QLabel(title)
            header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            header.setStyleSheet(f"color: {accent_color}; border: none; background: transparent;")
            outer_layout.addWidget(header)

            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet(f"background-color: {GRID_COL}; border: none; max-height: 1px;")
            outer_layout.addWidget(line)

        content_layout = QVBoxLayout()
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(0, 4, 0, 0)
        outer_layout.addLayout(content_layout)

        return card, content_layout

    def _build_controls(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340)
        scroll.setStyleSheet(
            f"QScrollArea{{ background: transparent; border:none; }}"
            f"QScrollBar:vertical{{ background: transparent; width:8px; border:none; }}"
            f"QScrollBar::handle:vertical{{ background:{GRID_COL}; border-radius:4px; }}"
        )

        w = QWidget()
        w.setStyleSheet(f"background: transparent;")
        layout = QVBoxLayout(w)
        layout.setSpacing(14)
        layout.setContentsMargins(0, 0, 10, 0) 

        title = QLabel("⚙️ Edge Detection Controls")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{TEXT_COL}; padding-bottom:4px;")
        layout.addWidget(title)

        card_m, cm_layout = self._create_card("🔍 Detection Method (Tab 4)", METHOD_COLORS['Prewitt'])
        self.method_combo = QComboBox()
        self.method_combo.addItems(list(KERNELS.keys()))
        self.method_combo.setStyleSheet(
            f"QComboBox {{ background:{BG_PANEL}; color:{TEXT_COL}; padding:8px 12px; font-size:12px; font-family:'Segoe UI'; border:1px solid {GRID_COL}; border-radius:6px; }}"
            f"QComboBox::drop-down {{ border:none; }}"
            f"QComboBox QAbstractItemView {{ background:{BG_PANEL}; color:{TEXT_COL}; selection-background-color:{STAGE_COLORS['gradient']}; selection-color:{BG_DARK}; outline:none; }}"
        )
        self.method_combo.currentTextChanged.connect(self._update)
        cm_layout.addWidget(self.method_combo)
        layout.addWidget(card_m)

        card_enh, cenh_layout = self._create_card("✨ Enhancement Method", STAGE_COLORS['enhancement'])
        self.enh_combo = QComboBox()
        self.enh_combo.addItems(["CLAHE", "Histogram Equalization", "Contrast Stretching", "None (Original)"])
        self.enh_combo.setStyleSheet(self.method_combo.styleSheet())
        self.enh_combo.currentTextChanged.connect(self._update)
        cenh_layout.addWidget(self.enh_combo)
        
        hint_enh = QLabel("Selects which method is passed down the pipeline.")
        hint_enh.setStyleSheet(f"color:{SUBTEXT}; font-size:11px; font-family:'Segoe UI';")
        hint_enh.setWordWrap(True)
        cenh_layout.addWidget(hint_enh)
        layout.addWidget(card_enh)

        card_s, cs_layout = self._create_card("Gaussian σ Restoration (Tab 3)", STAGE_COLORS['restoration'])
        self.sigma_lbl = QLabel("σ = 0.80")
        self.sigma_lbl.setStyleSheet(f"color:{STAGE_COLORS['restoration']}; font-weight:bold; font-size:12px; font-family:'Segoe UI';")
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(0, 50)
        self.sigma_slider.setValue(8)
        self.sigma_slider.setStyleSheet(
            f"QSlider::groove:horizontal{{ background:{GRID_COL}; height:6px; border-radius:3px; }}"
            f"QSlider::handle:horizontal{{ background:{STAGE_COLORS['restoration']}; width:16px; height:16px; margin:-5px 0; border-radius:8px; }}"
            f"QSlider::sub-page:horizontal{{ background:{STAGE_COLORS['restoration']}; border-radius:3px; }}"
        )
        self.sigma_slider.valueChanged.connect(self._sigma_changed)
        hint_s = QLabel("0 = no denoising  |  50 = strong blur")
        hint_s.setStyleSheet(f"color:{SUBTEXT}; font-size:11px; font-family:'Segoe UI';")
        cs_layout.addWidget(self.sigma_lbl)
        cs_layout.addWidget(self.sigma_slider)
        cs_layout.addWidget(hint_s)
        layout.addWidget(card_s)

        card_t, ct_layout = self._create_card("Edge Threshold Results (Tab 5)", STAGE_COLORS['results'])
        self.thr_lbl = QLabel("threshold = 0.12")
        self.thr_lbl.setStyleSheet(f"color:{STAGE_COLORS['results']}; font-weight:bold; font-size:12px; font-family:'Segoe UI';")
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(1, 60)
        self.thr_slider.setValue(12)
        self.thr_slider.setStyleSheet(
            f"QSlider::groove:horizontal{{ background:{GRID_COL}; height:6px; border-radius:3px; }}"
            f"QSlider::handle:horizontal{{ background:{STAGE_COLORS['results']}; width:16px; height:16px; margin:-5px 0; border-radius:8px; }}"
            f"QSlider::sub-page:horizontal{{ background:{STAGE_COLORS['results']}; border-radius:3px; }}"
        )
        self.thr_slider.valueChanged.connect(self._thr_changed)
        hint_t = QLabel("lower = more edges  |  higher = fewer")
        hint_t.setStyleSheet(f"color:{SUBTEXT}; font-size:11px; font-family:'Segoe UI';")
        ct_layout.addWidget(self.thr_lbl)
        ct_layout.addWidget(self.thr_slider)
        ct_layout.addWidget(hint_t)
        layout.addWidget(card_t)

        btn_load = QPushButton("📂  Load Facial Image")
        btn_load.setMinimumHeight(44)
        btn_load.setStyleSheet(
            f"QPushButton{{ background:{BG_CARD}; color:{TEXT_COL}; border:1px solid {GRID_COL}; border-radius:8px; font-weight:bold; font-size:13px; font-family:'Segoe UI'; }}"
            f"QPushButton:hover{{ background:{BG_PANEL}; border-color:{STAGE_COLORS['preprocessing']}; }}"
            f"QPushButton:pressed{{ background:{STAGE_COLORS['preprocessing']}; color:{BG_DARK}; }}"
        )
        btn_load.clicked.connect(self._load_image)
        layout.addWidget(btn_load)

        btn_save = QPushButton("💾  Save All Panels Now")
        btn_save.setMinimumHeight(44)
        btn_save.setStyleSheet(
            f"QPushButton{{ background:{BG_CARD}; color:{STAGE_COLORS['results']}; border:1px solid {STAGE_COLORS['results']}50; border-radius:8px; font-weight:bold; font-size:13px; font-family:'Segoe UI'; }}"
            f"QPushButton:hover{{ background:{BG_PANEL}; border-color:{STAGE_COLORS['results']}; }}"
            f"QPushButton:pressed{{ background:{STAGE_COLORS['results']}; color:{BG_DARK}; }}"
        )
        btn_save.clicked.connect(self._save_all_now)
        layout.addWidget(btn_save)

        card_stats, cstats_layout = self._create_card("📊 Live Metrics", TEXT_COL)
        self.stats_lbl = QLabel("Run detection to see metrics")
        self.stats_lbl.setFont(QFont("Consolas", 11))
        self.stats_lbl.setStyleSheet(f"color:{SUBTEXT}; line-height: 1.5;")
        self.stats_lbl.setWordWrap(True)
        cstats_layout.addWidget(self.stats_lbl)
        layout.addWidget(card_stats)

        card_info, cinfo_layout = self._create_card("📐 Kernel Info", SUBTEXT)
        self.info_lbl = QLabel("")
        self.info_lbl.setFont(QFont("Consolas", 10))
        self.info_lbl.setStyleSheet(f"color:{SUBTEXT};")
        self.info_lbl.setWordWrap(True)
        cinfo_layout.addWidget(self.info_lbl)
        layout.addWidget(card_info)

        self.autosave_lbl = QLabel("💾 Auto-save: enabled (3 s delay)")
        self.autosave_lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:11px; padding-left:4px; font-family:'Segoe UI';")
        self.autosave_lbl.setWordWrap(True)
        layout.addWidget(self.autosave_lbl)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_tabbed_view(self) -> QTabWidget:
        # Tab Utama (Main Wrapper)
        main_tabs = QTabWidget()
        main_tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: none; background: transparent; }}"
            f"QTabBar::tab {{ background: {BG_DARK}; color: {SUBTEXT}; padding: 12px 20px; font-size: 14px; font-weight: bold; margin-right: 5px; border-radius: 8px; }}"
            f"QTabBar::tab:selected {{ background: {BG_PANEL}; color: {TEXT_COL}; }}"
            f"QTabBar::tab:hover {{ background: {BG_CARD}; color: #ffffff; }}"
        )

        sub_tab_style = (
            f"QTabWidget::pane {{ border:1px solid {GRID_COL}; border-radius: 10px; background: rgba(27, 40, 56, 0.45); padding: 4px; }}" 
            f"QTabBar::tab {{ background:{BG_DARK}; color:{SUBTEXT}; padding:10px 18px; margin-right: 4px; margin-bottom: 8px; border:1px solid {GRID_COL}; border-radius:8px; font-family:'Segoe UI'; font-size:12px; font-weight:bold; }}"
            f"QTabBar::tab:selected {{ background:{BG_CARD}; color:{TEXT_COL}; border:1px solid {STAGE_COLORS['preprocessing']}; }}"
            f"QTabBar::tab:hover {{ background:{BG_CARD}; color:{TEXT_COL}; }}"
        )

        # --- Bungkus Sub-Tab Assignment 1 ---
        tabs_a1 = QTabWidget()
        tabs_a1.setStyleSheet(sub_tab_style)
        tabs_a1.addTab(self._build_tab_acquisition(),          " 1. Acquisition ")
        tabs_a1.addTab(self._build_tab_enhancement(),          " 2. Image Enhancement ")
        tabs_a1.addTab(self._build_tab_restoration(),          " 3. Restoration ")
        tabs_a1.addTab(self._build_tab_gradient(),             " 4. Gradient and Detection ")
        tabs_a1.addTab(self._build_tab_results(),              " 5. Results and Analysis ")

        # --- Bungkus Sub-Tab Assignment 2 ---
        tabs_a2 = QTabWidget()
        tabs_a2.setStyleSheet(sub_tab_style.replace(STAGE_COLORS['preprocessing'], STAGE_COLORS['canny']))
        tabs_a2.addTab(self._build_tab_canny(),                " 6. Canny Edge Detection ")
        tabs_a2.addTab(self._build_tab_sharpening(),           " 7. Image Sharpening ")
        tabs_a2.addTab(self._build_tab_runtime_comparison(),   " 8. Runtime and Comparison ")

        # --- Masukkan Kedua Sub-Tab ke Main Tab ---
        main_tabs.addTab(tabs_a1, " 📚 ASSIGNMENT 1 ")
        main_tabs.addTab(tabs_a2, " 📚 ASSIGNMENT 2 ")
        main_tabs.addTab(self._build_tab_secret(), " ✧ ")

        # Event listeners
        main_tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs_a2 = tabs_a2
        self.tabs_a2.currentChanged.connect(self._on_sub_tab_changed)

        return main_tabs

    def _build_tab_secret(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent; border: none;")
        return w

    def _on_tab_changed(self, index: int):
        if not hasattr(self, 'left_panel') or not hasattr(self, 'status') or not hasattr(self, 'bg_widget'):
            return

        if index == 2:   # Secret tab sekarang ada di index ke-2
            self.left_panel.hide()
            self.status.hide()
            self.bg_widget.opacity = 1.0
            self.bg_widget.dimness = 0.0
            self.bg_widget.update()
        else:
            self.left_panel.show()
            self.status.show()
            self.bg_widget.opacity = 0.45
            self.bg_widget.dimness = 0.60
            self.bg_widget.update()
            
            # Auto-update Runtime ketika membuka Assignment 2 jika diperlukan
            if index == 1 and hasattr(self, 'tabs_a2') and self.tabs_a2.currentIndex() == 2:
                if getattr(self, '_runtime_dirty', False):
                    QTimer.singleShot(150, self._do_update_runtime)
    
    def _on_sub_tab_changed(self, index: int):
        # Auto-update ketika berpindah di dalam sub-tab Assignment 2
        if index == 2 and getattr(self, '_runtime_dirty', False):
            QTimer.singleShot(150, self._do_update_runtime)

    def _scroll_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea{{ background: transparent; border:none; }}"
            f"QScrollBar:vertical  {{ background: transparent; width:10px; border:none; }}"
            f"QScrollBar:horizontal{{ background: transparent; height:10px; border:none; }}"
            f"QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ background:{GRID_COL}; border-radius:5px; }}"
        )
        content = QWidget()
        content.setStyleSheet(f"background: transparent;")
        layout = QVBoxLayout(content)
        layout.setSpacing(16)
        layout.setContentsMargins(12, 12, 12, 12)
        scroll.setWidget(content)
        return scroll, layout

    def _section_header(self, stage: str, desc: str, color: str) -> QWidget:
        w = QWidget()
        w.setFixedHeight(48)
        w.setStyleSheet(
            f"background:{BG_CARD}; border-radius:8px;"
            f"border-left:6px solid {color};"
        )
        l = QHBoxLayout(w)
        l.setContentsMargins(16, 4, 16, 4)
        l.setSpacing(14)

        s_lbl = QLabel(stage)
        s_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        s_lbl.setStyleSheet(f"color:{color}; background:transparent; border:none;")
        
        sep = QLabel("│")
        sep.setStyleSheet(f"color:{GRID_COL}; background:transparent; border:none; font-size:16px;")
        
        d_lbl = QLabel(desc)
        d_lbl.setFont(QFont("Segoe UI", 11))
        d_lbl.setStyleSheet(f"color:{SUBTEXT}; background:transparent; border:none;")

        l.addWidget(s_lbl); l.addWidget(sep); l.addWidget(d_lbl)
        l.addStretch()
        return w

    def _panel_row(self, widgets: list) -> QWidget:
        row = QWidget()
        row.setStyleSheet(f"background:transparent;")
        rl = QHBoxLayout(row)
        rl.setSpacing(12)
        rl.setContentsMargins(0, 0, 0, 0)
        for wgt in widgets:
            rl.addWidget(wgt)
        return row

    def _slider_card(self, label_text, range_lo, range_hi, default_val, unit="", color="#89B4FA"):
        card, layout = self._create_card(label_text, color)
        
        val_lbl = QLabel(f"{default_val}{unit}")
        val_lbl.setStyleSheet(f"color:{color}; font-weight:bold; font-size:13px; font-family:'Segoe UI';")
        
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(range_lo, range_hi)
        sld.setValue(default_val)
        sld.setStyleSheet(
            f"QSlider::groove:horizontal{{ background:{GRID_COL}; height:6px; border-radius:3px; }}"
            f"QSlider::handle:horizontal{{ background:{color}; width:16px; height:16px; margin:-5px 0; border-radius:8px; }}"
            f"QSlider::sub-page:horizontal{{ background:{color}; border-radius:3px; }}"
        )
        layout.addWidget(val_lbl)
        layout.addWidget(sld)
        return card, sld, val_lbl

    def _build_tab_acquisition(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "1. ACQUISITION", "Original Grayscale → np.clip [0,1] → Acquired Image", STAGE_COLORS["preprocessing"]))

        self.panels["acq_orig"] = ImagePanel("Original Gray", "As loaded / rgb2gray", "preprocessing")
        self.panels["acq"] = ImagePanel("1. Acquired", "np.clip(gray, 0, 1)", "preprocessing")
        layout.addWidget(self._panel_row([self.panels["acq_orig"], self.panels["acq"]]))

        layout.addWidget(self._section_header(
            "  ANALYSIS", "Histogram and Ogive (CDF) Original vs Acquired", STAGE_COLORS["preprocessing"]))

        self.analysis_panels["acq_orig_ha"] = AnalysisPanel(
            "Histogram + Ogive", "Original Gray", "preprocessing", figsize=(4.5, 3.2))
        self.analysis_panels["acq_ha"] = AnalysisPanel(
            "Histogram + Ogive", "Acquired Image", "preprocessing", figsize=(4.5, 3.2))
        self.analysis_panels["acq_compare_ha"] = AnalysisPanel(
            "Comparison Overlay", "Original vs Acquired", "preprocessing", figsize=(5.0, 3.2))
        
        layout.addWidget(self._panel_row([
            self.analysis_panels["acq_orig_ha"],
            self.analysis_panels["acq_ha"],
            self.analysis_panels["acq_compare_ha"],
        ]))

        layout.addStretch()
        return scroll

    def _build_tab_enhancement(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "2. IMAGE ENHANCEMENT CONTROLS", "Adjust parameters for each method (CS, HE, CLAHE)", STAGE_COLORS["enhancement"]))

        ctrl_row = QWidget()
        ctrl_row.setStyleSheet(f"background:transparent;")
        ctrl_l = QHBoxLayout(ctrl_row)
        ctrl_l.setSpacing(14)
        ctrl_l.setContentsMargins(0, 0, 0, 0)

        card_cl, self.enh_clahe_sld, self.enh_clahe_lbl = self._slider_card(
            "CLAHE clip_limit (×0.01)", 1, 10, 3, unit=" → 0.03", color=ENH_COLORS["CLAHE"])
        self.enh_clahe_sld.valueChanged.connect(self._enh_clahe_changed)
        ctrl_l.addWidget(card_cl)

        card_cslo, self.enh_cs_low_sld, self.enh_cs_low_lbl = self._slider_card(
            "CS p_low (%)", 0, 15, 2, unit="%", color=ENH_COLORS["CS"])
        self.enh_cs_low_sld.valueChanged.connect(self._enh_cs_low_changed)
        ctrl_l.addWidget(card_cslo)

        card_cshi, self.enh_cs_high_sld, self.enh_cs_high_lbl = self._slider_card(
            "CS p_high (%)", 80, 100, 98, unit="%", color=ENH_COLORS["CS"])
        self.enh_cs_high_sld.valueChanged.connect(self._enh_cs_high_changed)
        ctrl_l.addWidget(card_cshi)

        info_lbl = QLabel(
            "<b>CS</b> = Contrast Stretching<br>"
            "<b>HE</b> = Histogram Equalization<br>"
            "<b>CLAHE</b> = Adaptive HE<br><br>"
            "<i>Modifying these will auto-update the main pipeline if selected.</i>")
        info_lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:12px; font-family:'Segoe UI'; padding: 12px; background-color:{BG_CARD}; border-radius:10px; border:1px solid {GRID_COL};")
        info_lbl.setWordWrap(True)
        ctrl_l.addWidget(info_lbl, 1)
        layout.addWidget(ctrl_row)

        layout.addWidget(self._section_header(
            "  ENHANCED IMAGES", "CS | Histogram Equalization | CLAHE side-by-side comparison", STAGE_COLORS["enhancement"]))

        self.panels["enh_cs"]    = ImagePanel("Contrast Stretching", "Percentile linear", "enhancement")
        self.panels["enh_he"]    = ImagePanel("Histogram Equalization", "Global HE", "enhancement")
        self.panels["enh_clahe"] = ImagePanel("CLAHE", "Adaptive HE", "enhancement")
        layout.addWidget(self._panel_row([self.panels["enh_cs"], self.panels["enh_he"], self.panels["enh_clahe"]]))

        layout.addWidget(self._section_header(
            "  HISTOGRAM and OGIVE PER METHOD", "Pixel intensity distribution + cumulative distribution function", STAGE_COLORS["enhancement"]))

        self.analysis_panels["enh_cs_ha"]    = AnalysisPanel("CS Hist + Ogive", "Contrast Stretching", "enhancement", figsize=(4.0, 3.2))
        self.analysis_panels["enh_he_ha"]    = AnalysisPanel("HE Hist + Ogive", "Histogram Equalization", "enhancement", figsize=(4.0, 3.2))
        self.analysis_panels["enh_clahe_ha"] = AnalysisPanel("CLAHE Hist + Ogive", "Adaptive HE", "enhancement", figsize=(4.0, 3.2))
        layout.addWidget(self._panel_row([
            self.analysis_panels["enh_cs_ha"],
            self.analysis_panels["enh_he_ha"],
            self.analysis_panels["enh_clahe_ha"],
        ]))

        layout.addWidget(self._section_header(
            "  PERFORMANCE EVALUATION", "RMSE · PSNR · SSIM · Shannon Entropy (reference = acquired image)", STAGE_COLORS["enhancement"]))

        self.analysis_panels["enh_perf"] = AnalysisPanel(
            "Performance Evaluation", "RMSE / PSNR / SSIM / Shannon Entropy", "enhancement", figsize=(10.0, 4.0))
        self.analysis_panels["enh_perf"].setMinimumHeight(350)
        layout.addWidget(self.analysis_panels["enh_perf"])

        layout.addStretch()
        return scroll

    def _build_tab_restoration(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "3. RESTORATION and MORPHO PRE-PROC", "Gaussian Denoising (σ slider in left panel) applied on selected Enhancement → Morphological Closing", STAGE_COLORS["restoration"]))

        self.panels["denoised"]   = ImagePanel("3. Restoration",  "Gaussian σ-controlled", "restoration")
        self.panels["morpho_pre"] = ImagePanel("4. Morpho Pre-proc", "Structural Closing disk(1)", "restoration")
        layout.addWidget(self._panel_row([self.panels["denoised"], self.panels["morpho_pre"]]))

        layout.addStretch()
        return scroll

    def _build_tab_gradient(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "4. GRADIENT and EDGE DETECTION", "Kernel Convolution → Gradient Gx and Gy → Magnitude √(Gx²+Gy²) → Direction arctan2", STAGE_COLORS["gradient"]))

        self.panels["gx"]        = ImagePanel("5. Gradient Gx",   "Horizontal Edges", "gradient")
        self.panels["gy"]        = ImagePanel("6. Gradient Gy",   "Vertical Edges",   "gradient")
        self.panels["magnitude"] = ImagePanel("7. Edge Magnitude","√(Gx²+Gy²) Norm.", "gradient")
        self.panels["direction"] = ImagePanel("8. Direction Map", "arctan2(Gy, Gx)",  "gradient")
        layout.addWidget(self._panel_row([
            self.panels["gx"], self.panels["gy"], self.panels["magnitude"], self.panels["direction"]
        ]))

        layout.addStretch()
        return scroll

    def _build_tab_results(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "5. RESULTS and ANALYSIS", "Thresholding → Morpho Thinning (Skeleton) → Magnitude Distribution → Row Profile", STAGE_COLORS["results"]))

        self.panels["binary"]      = ImagePanel("9. Thresholded",    "Binary Edge Map",      "results")
        self.panels["morpho_post"] = ImagePanel("10. Morpho Post-proc","Skeletonize / Thin",  "results")
        layout.addWidget(self._panel_row([self.panels["binary"], self.panels["morpho_post"]]))

        layout.addWidget(self._section_header(
            "  MAGNITUDE ANALYSIS", "Histogram (with threshold marker) · Histogram + Ogive (CDF) · Row Profile", STAGE_COLORS["results"]))

        self.panels["histogram"] = ImagePanel("11. Magnitude Hist.", "Distribution + threshold marker", "results")
        self.panels["profile"]   = ImagePanel("12. Row Profile",     "Intensity Cross-section",         "results")
        self.analysis_panels["result_mag_ha"] = AnalysisPanel("Magnitude Hist + Ogive", "Edge magnitude CDF for threshold selection", "results", figsize=(4.5, 3.2))

        layout.addWidget(self._panel_row([
            self.panels["histogram"],
            self.analysis_panels["result_mag_ha"],
            self.panels["profile"],
        ]))

        layout.addStretch()
        return scroll

    # TAB 6: CANNY EDGE DETECTION
    def _build_tab_canny(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "6. CANNY EDGE DETECTION",
            "Manual from-scratch implementation + skimage library comparison",
            STAGE_COLORS["canny"]))

        # Inline parameter controls
        ctrl = QWidget()
        ctrl.setStyleSheet("background:transparent;")
        cl = QHBoxLayout(ctrl)
        cl.setSpacing(14); cl.setContentsMargins(0,0,0,0)

        card_s, self.canny_sigma_sld, self.canny_sigma_lbl = self._slider_card(
            "Gaussian σ (×0.1)", 1, 30, 10, unit=" → 1.0", color=STAGE_COLORS["canny"])
        self.canny_sigma_sld.valueChanged.connect(self._canny_sigma_changed)
        cl.addWidget(card_s)

        card_lo, self.canny_tlo_sld, self.canny_tlo_lbl = self._slider_card(
            "Low Threshold (×0.01)", 1, 30, 5, unit=" → 0.05", color="#FAB387")
        self.canny_tlo_sld.valueChanged.connect(self._canny_tlo_changed)
        cl.addWidget(card_lo)

        card_hi, self.canny_thi_sld, self.canny_thi_lbl = self._slider_card(
            "High Threshold (×0.01)", 5, 50, 15, unit=" → 0.15", color="#A6E3A1")
        self.canny_thi_sld.valueChanged.connect(self._canny_thi_changed)
        cl.addWidget(card_hi)

        info_c = QLabel(
            "<b>Pre-processing:</b> Gaussian smoothing → Gaussian gradient (fx, fy)<br>"
            "<b>Edge Localization:</b> Magnitude → Direction → Digitize angle → NMS<br>"
            "<b>Hysteresis:</b> Double threshold (Lo=weak, Hi=strong) → edge tracking")
        info_c.setStyleSheet(
            f"color:{SUBTEXT}; font-size:12px; font-family:'Segoe UI';"
            f" padding:12px; background:{BG_CARD}; border-radius:10px;"
            f" border:1px solid {GRID_COL};")
        info_c.setWordWrap(True)
        cl.addWidget(info_c, 1)
        layout.addWidget(ctrl)

        # Pre-processing row
        layout.addWidget(self._section_header(
            "  PRE-PROCESSING: Gaussian Smoothing + Gaussian Gradient",
            "Ī = I * H^{G,σ}  →  Īx = ∂H/∂x   Īy = ∂H/∂y",
            STAGE_COLORS["canny"]))

        self.panels["canny_smooth"] = ImagePanel("Smoothed (Ī)",     "Gaussian σ", "canny")
        self.panels["canny_fx"]     = ImagePanel("Gradient fx (Īx)", "∂H^{G,σ}/∂x", "canny")
        self.panels["canny_fy"]     = ImagePanel("Gradient fy (Īy)", "∂H^{G,σ}/∂y", "canny")
        self.panels["canny_mag"]    = ImagePanel("Magnitude (Emag)", "√(fx²+fy²)", "canny")
        layout.addWidget(self._panel_row([
            self.panels["canny_smooth"], self.panels["canny_fx"],
            self.panels["canny_fy"],     self.panels["canny_mag"],
        ]))

        # Edge Localization + Hysteresis row
        layout.addWidget(self._section_header(
            "  EDGE LOCALIZATION + HYSTERESIS THRESHOLDING",
            "Φ(u,v)=arctan2(Iy,Ix) → Digitize → NMS → Double threshold → Hysteresis",
            STAGE_COLORS["canny"]))

        self.panels["canny_angle"]  = ImagePanel("Direction Φ(u,v)",    "arctan2(Iy,Ix) [HSV]", "canny")
        self.panels["canny_quant"]  = ImagePanel("Digitized Direction",  "0=H  1=↗  2=V  3=↘",  "canny")
        self.analysis_panels["canny_color"] = AnalysisPanel(
            "Color Direction", "R=horiz  G=diag↗  B=vert  Y=diag↘", "canny", figsize=(3.5, 3.0))
        self.panels["canny_nms"]    = ImagePanel("Non-Max Suppressed",   "Enms",                 "canny")
        self.panels["canny_thresh"] = ImagePanel("Double Threshold",     "White=strong  Grey=weak","canny")
        self.panels["canny_hys"]    = ImagePanel("Hysteresis (Final)",   "Edge map",              "canny")
        layout.addWidget(self._panel_row([
            self.panels["canny_angle"],
            self.panels["canny_quant"],
            self.analysis_panels["canny_color"],
            self.panels["canny_nms"],
            self.panels["canny_thresh"],
            self.panels["canny_hys"],
        ]))

        # Library Canny
        layout.addWidget(self._section_header(
            "  CANNY (LIBRARY: skimage.feature.canny)",
            "Same σ / lo / hi thresholds applied via optimised library implementation",
            STAGE_COLORS["canny"]))

        self.panels["canny_lib"]  = ImagePanel("Canny Library",    "skimage.feature.canny", "canny")
        self.analysis_panels["canny_compare"] = AnalysisPanel(
            "Scratch vs Library", "Edge pixel distribution comparison",
            "canny", figsize=(6.0, 3.2))
        layout.addWidget(self._panel_row([
            self.panels["canny_lib"],
            self.analysis_panels["canny_compare"],
        ]))

        layout.addStretch()
        return scroll
    
    # TAB 7: IMAGE SHARPENING
    def _build_tab_sharpening(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "7. IMAGE SHARPENING",
            "Laplacian Operator  I' = I − w·(H^L * I)   ·   Unsharp Masking  I' = I + a·(I − blur)",
            STAGE_COLORS["sharpening"]))

        # Controls
        ctrl = QWidget()
        ctrl.setStyleSheet("background:transparent;")
        cl = QHBoxLayout(ctrl)
        cl.setSpacing(14); cl.setContentsMargins(0,0,0,0)

        card_w, self.lap_weight_sld, self.lap_weight_lbl = self._slider_card(
            "Laplacian weight w (×0.1)", 1, 30, 10, unit=" → 1.0", color=STAGE_COLORS["sharpening"])
        self.lap_weight_sld.valueChanged.connect(self._lap_weight_changed)
        cl.addWidget(card_w)

        # Kernel selector (QComboBox inside a card)
        kern_card, kern_layout = self._create_card("Laplacian Kernel", STAGE_COLORS["sharpening"])
        self.lap_kernel_combo = QComboBox()
        self.lap_kernel_combo.addItems(["H4  (4-conn  |  0,1,0 / 1,-4,1 / 0,1,0)",
                                        "H8  (8-conn  |  1,1,1 / 1,-8,1 / 1,1,1)",
                                        "H12 (weighted|  1,2,1 / 2,-12,2 / 1,2,1)"])
        self.lap_kernel_combo.setStyleSheet(
            f"QComboBox {{background:{BG_PANEL}; color:{TEXT_COL}; padding:8px 12px; font-size:11px;"
            f" border:1px solid {GRID_COL}; border-radius:6px; }}"
            f"QComboBox QAbstractItemView {{background:{BG_PANEL}; color:{TEXT_COL}; }}")
        self.lap_kernel_combo.currentTextChanged.connect(self._lap_kernel_changed)
        kern_layout.addWidget(self.lap_kernel_combo)
        cl.addWidget(kern_card)

        card_a, self.usm_a_sld, self.usm_a_lbl = self._slider_card(
            "USM factor a (×0.1)", 1, 30, 7, unit=" → 0.7", color="#89B4FA")
        self.usm_a_sld.valueChanged.connect(self._usm_a_changed)
        cl.addWidget(card_a)

        card_us, self.usm_sigma_sld, self.usm_sigma_lbl = self._slider_card(
            "USM Gaussian σ (×0.1)", 1, 30, 10, unit=" → 1.0", color="#89B4FA")
        self.usm_sigma_sld.valueChanged.connect(self._usm_sigma_changed)
        cl.addWidget(card_us)

        layout.addWidget(ctrl)

        # Laplacian section
        layout.addWidget(self._section_header(
            "  LAPLACIAN SHARPENING",
            "Separable: H_x=[1,-2,1]  H_y=[[1],[-2],[1]]  →  Full 2D kernel H^L",
            STAGE_COLORS["sharpening"]))

        self.panels["sharp_orig"]     = ImagePanel("Input (Gray)",       "Original grayscale",   "sharpening")
        self.panels["sharp_lap_x"]    = ImagePanel("Laplacian X",        "|H_x * blur|",         "sharpening")
        self.panels["sharp_lap_y"]    = ImagePanel("Laplacian Y",        "|H_y * blur|",         "sharpening")
        self.panels["sharp_lap_sum"]  = ImagePanel("Lap XY Sum",         "|Lap_x + Lap_y|",      "sharpening")
        self.panels["sharp_sep"]      = ImagePanel("Sharpened (Sep.)",   "I − w·(Lap_x+Lap_y)", "sharpening")
        layout.addWidget(self._panel_row([
            self.panels["sharp_orig"],    self.panels["sharp_lap_x"],
            self.panels["sharp_lap_y"],   self.panels["sharp_lap_sum"],
            self.panels["sharp_sep"],
        ]))

        self.panels["sharp_lap_full"] = ImagePanel("Laplacian H^L",      "|H^L * blur|",         "sharpening")
        self.panels["sharp_full"]     = ImagePanel("Sharpened (Full)",   "I − w·(H^L*I)",       "sharpening")
        layout.addWidget(self._panel_row([
            self.panels["sharp_lap_full"], self.panels["sharp_full"],
        ]))

        # Unsharp Masking section
        layout.addWidget(self._section_header(
            "  UNSHARP MASKING (USM)",
            "M = I − blur(I, σ)   →   I' = I + a · M",
            STAGE_COLORS["sharpening"]))

        self.panels["sharp_usm_blur"]   = ImagePanel("USM Blurred",     "Gaussian blur", "sharpening")
        self.panels["sharp_usm_mask"]   = ImagePanel("USM Mask  M",     "I − blur",      "sharpening")
        self.panels["sharp_usm_result"] = ImagePanel("USM Sharpened I'","I + a·M",       "sharpening")
        layout.addWidget(self._panel_row([
            self.panels["sharp_orig"],           # reuse the same panel reference
            self.panels["sharp_usm_blur"],
            self.panels["sharp_usm_mask"],
            self.panels["sharp_usm_result"],
        ]))

        layout.addStretch()
        return scroll

    # TAB 8: RUNTIME & COMPARISON
    def _build_tab_runtime_comparison(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "8. RUNTIME and METHOD COMPARISON",
            "Run all methods → measure elapsed time → compare edge outputs side-by-side",
            STAGE_COLORS["comparison"]))

        # Button to force re-run
        btn_run = QPushButton("▶  Run All Methods Now")
        btn_run.setMinimumHeight(44)
        btn_run.setStyleSheet(
            f"QPushButton {{ background:{BG_CARD}; color:{STAGE_COLORS['comparison']};"
            f" border:1px solid {STAGE_COLORS['comparison']}60; border-radius:8px;"
            f" font-weight:bold; font-size:13px; font-family:'Segoe UI'; }}"
            f"QPushButton:hover {{ background:{BG_PANEL}; border-color:{STAGE_COLORS['comparison']}; }}"
            f"QPushButton:pressed {{ background:{STAGE_COLORS['comparison']}; color:{BG_DARK}; }}")
        btn_run.clicked.connect(self._do_update_runtime)
        layout.addWidget(btn_run)

        # Runtime bar chart
        layout.addWidget(self._section_header(
            "  RUNTIME ANALYSIS",
            "Elapsed time per method (ms) — green=fast  orange=medium  red=slow",
            STAGE_COLORS["comparison"]))

        self.analysis_panels["runtime_chart"] = AnalysisPanel(
            "Runtime Comparison", "All 9 methods benchmarked on current image",
            "comparison", figsize=(12.0, 4.5))
        self.analysis_panels["runtime_chart"].setMinimumHeight(360)
        layout.addWidget(self.analysis_panels["runtime_chart"])

        # Edge detection grid
        layout.addWidget(self._section_header(
            "  EDGE DETECTION COMPARISON",
            "Prewitt · Sobel · Roberts · Ext. Sobel · Kirsch · Canny Scratch · Canny Library",
            STAGE_COLORS["comparison"]))

        self.analysis_panels["edge_compare_grid"] = AnalysisPanel(
            "Edge Detection Grid", "7 methods side-by-side",
            "comparison", figsize=(14.0, 7.0))
        self.analysis_panels["edge_compare_grid"].setMinimumHeight(480)
        layout.addWidget(self.analysis_panels["edge_compare_grid"])

        # Sharpening comparison
        layout.addWidget(self._section_header(
            "  SHARPENING COMPARISON",
            "Original  ·  Laplacian Sharpened  ·  Unsharp Masking Result",
            STAGE_COLORS["comparison"]))

        self.analysis_panels["sharp_compare_grid"] = AnalysisPanel(
            "Sharpening Grid", "Original vs Laplacian vs USM",
            "comparison", figsize=(10.0, 4.0))
        self.analysis_panels["sharp_compare_grid"].setMinimumHeight(320)
        layout.addWidget(self.analysis_panels["sharp_compare_grid"])

        layout.addStretch()
        return scroll

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Facial Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)")
        if not path:
            return
        try:
            if HAS_CV2:
                bgr = cv2.imread(path)
                if bgr is not None:
                    rgb = cv2.cvtColor(cv2.resize(bgr, (300, 300)), cv2.COLOR_BGR2RGB)
                    self.gray_img = manual_rgb2gray(rgb)
            else:
                from PIL import Image as PILImage
                pil = PILImage.open(path).convert("RGB").resize((300, 300))
                self.gray_img = manual_rgb2gray(np.array(pil))

            self.image_name = os.path.basename(path)
            # Reset caches so new image re-triggers computation
            self._last_canny_scratch  = None
            self._last_canny_lib      = None
            self._last_sharpening_lap = None
            self._last_sharpening_usm = None
            self._runtime_dirty       = True

            self._update()
            # Trigger Canny + Sharpening after pipeline settles
            self._canny_debounce.start(600)
            self._sharp_debounce.start(600)
            self.status.showMessage(f"Loaded: {path}", 5000)
        except Exception as e:
            self.status.showMessage(f"Error loading image: {e}", 5000)

    def _sigma_changed(self, val):
        sigma = val / 10.0
        self.sigma_lbl.setText(f"σ = {sigma:.2f}")
        self._update()

    def _thr_changed(self, val):
        thr = val / 100.0
        self.thr_lbl.setText(f"threshold = {thr:.2f}")
        self._update()

    def _enh_clahe_changed(self, val: int):
        self.enh_clahe_clip = val / 100.0
        self.enh_clahe_lbl.setText(f"clip_limit = {self.enh_clahe_clip:.2f}")
        self._update()

    def _enh_cs_low_changed(self, val: int):
        self.cs_low = float(val)
        self.enh_cs_low_lbl.setText(f"p_low = {val}%")
        self._update()

    def _enh_cs_high_changed(self, val: int):
        self.cs_high = float(val)
        self.enh_cs_high_lbl.setText(f"p_high = {val}%")
        self._update()

    def _update(self):
        self._debounce.start(180)

    # Slider callbacks
    def _canny_sigma_changed(self, val: int):
        self.canny_sigma = val / 10.0
        self.canny_sigma_lbl.setText(f"σ = {self.canny_sigma:.1f}")
        self._canny_debounce.start(350)

    def _canny_tlo_changed(self, val: int):
        self.canny_t_lo = val / 100.0
        self.canny_tlo_lbl.setText(f"lo = {self.canny_t_lo:.2f}")
        self._canny_debounce.start(350)

    def _canny_thi_changed(self, val: int):
        self.canny_t_hi = val / 100.0
        self.canny_thi_lbl.setText(f"hi = {self.canny_t_hi:.2f}")
        self._canny_debounce.start(350)

    def _lap_weight_changed(self, val: int):
        self.lap_weight = val / 10.0
        self.lap_weight_lbl.setText(f"w = {self.lap_weight:.1f}")
        self._sharp_debounce.start(350)

    def _lap_kernel_changed(self, text: str):
        if "H8"  in text: self.lap_kernel = "H8"
        elif "H12" in text: self.lap_kernel = "H12"
        else:              self.lap_kernel = "H4"
        self._sharp_debounce.start(350)

    def _usm_a_changed(self, val: int):
        self.usm_a = val / 10.0
        self.usm_a_lbl.setText(f"a = {self.usm_a:.1f}")
        self._sharp_debounce.start(350)

    def _usm_sigma_changed(self, val: int):
        self.usm_sigma = val / 10.0
        self.usm_sigma_lbl.setText(f"σ = {self.usm_sigma:.1f}")
        self._sharp_debounce.start(350)

    # Update methods
    def _do_update_canny(self):
        """Compute Canny from scratch + library and refresh Tab 6 panels."""
        if self.gray_img is None:
            return
        try:
            gray = self.gray_img
            # From scratch
            sc = canny_scratch(gray, self.canny_sigma, self.canny_t_lo, self.canny_t_hi)
            self._last_canny_scratch = sc

            self.panels["canny_smooth"].show_image(sc["smoothed"],       "gray")
            self.panels["canny_fx"].show_image(sc["fx"],                 "RdBu_r")
            self.panels["canny_fy"].show_image(sc["fy"],                 "PRGn_r")
            self.panels["canny_mag"].show_image(sc["magnitude"],         EDGE_CMAP, colorbar=True)
            self.panels["canny_angle"].show_image(sc["angle_disp"],      "hsv",     colorbar=True)
            self.panels["canny_quant"].show_image(sc["quantized_disp"],  CANNY_DIR_CMAP)
            self.analysis_panels["canny_color"].show_rgb_image(
                sc["color_rgb"],
                title="Direction Colors  R=Horiz  G=Diag↗  B=Vert  Y=Diag↘")
            self.panels["canny_nms"].show_image(sc["nms"],               EDGE_CMAP, colorbar=True)
            self.panels["canny_thresh"].show_image(sc["double_thresh_disp"], "gray")
            self.panels["canny_hys"].show_image(sc["hysteresis"],        "gray")

            # Library
            lib = canny_library(gray, self.canny_sigma, self.canny_t_lo, self.canny_t_hi)
            self._last_canny_lib = lib

            self.panels["canny_lib"].show_image(lib["result"], "gray")

            # Comparison histogram overlay
            self.analysis_panels["canny_compare"].show_comparison_hist_ogive(
                {"Scratch": sc["hysteresis"], "Library": lib["result"]},
                title=f"Canny Comparison  |  Scratch: {sc['elapsed']:.1f} ms  "
                      f"|  Library: {lib['elapsed']:.1f} ms  "
                      f"|  Scratch density: {sc['density']:.4f}  "
                      f"|  Library density: {lib['density']:.4f}")

            self._runtime_dirty = True
        except Exception as e:
            self.status.showMessage(f"[Canny error] {e}", 4000)

    def _do_update_sharpening(self):
        """Compute Laplacian + USM sharpening and refresh Tab 7 panels."""
        if self.gray_img is None:
            return
        try:
            gray = self.gray_img
            # Laplacian
            lap = laplacian_sharpening(gray, self.lap_weight, self.lap_kernel)
            self._last_sharpening_lap = lap

            self.panels["sharp_orig"].show_image(gray,               "gray")
            self.panels["sharp_lap_x"].show_image(lap["lap_x"],      EDGE_CMAP)
            self.panels["sharp_lap_y"].show_image(lap["lap_y"],      EDGE_CMAP)
            self.panels["sharp_lap_sum"].show_image(lap["lap_xy_sep"], EDGE_CMAP)
            self.panels["sharp_sep"].show_image(lap["sharp_sep"],    "gray")
            self.panels["sharp_lap_full"].show_image(lap["lap_full"], EDGE_CMAP)
            self.panels["sharp_full"].show_image(lap["sharp_full"],  "gray")

            # Unsharp Masking
            usm = unsharp_masking(gray, self.usm_a, self.usm_sigma)
            self._last_sharpening_usm = usm

            self.panels["sharp_usm_blur"].show_image(usm["blurred"],   "gray")
            self.panels["sharp_usm_mask"].show_image(usm["mask"],      "RdBu_r")
            self.panels["sharp_usm_result"].show_image(usm["sharpened"], "gray")

            self._runtime_dirty = True
        except Exception as e:
            self.status.showMessage(f"[Sharpening error] {e}", 4000)

    def _do_update_runtime(self):
        """Run ALL methods, measure runtimes, update Tab 8 panels."""
        if self.gray_img is None:
            return
        try:
            self.status.showMessage("⏳  Running all methods for comparison...", 0)
            QApplication.processEvents()

            sigma   = self.sigma_slider.value() / 10.0
            thr     = self.thr_slider.value() / 100.0
            enh_txt = self.enh_combo.currentText()
            if "CLAHE" in enh_txt:     enh_m = "CLAHE"
            elif "Histogram" in enh_txt: enh_m = "HE"
            elif "Contrast" in enh_txt:  enh_m = "CS"
            else:                        enh_m = "None"

            results = compute_all_methods(
                self.gray_img, sigma, thr,
                self.canny_sigma, self.canny_t_lo, self.canny_t_hi,
                self.lap_weight, self.lap_kernel,
                self.usm_a, self.usm_sigma,
                enh_m, self.enh_clahe_clip, self.cs_low, self.cs_high)

            # Runtime bar chart
            runtimes = {m: results[m]["elapsed"] for m in results}
            self._last_runtimes = runtimes
            self.analysis_panels["runtime_chart"].show_runtime_bars(runtimes)

            # Edge detection comparison grid (7 edge methods)
            edge_keys = ["Prewitt", "Sobel", "Roberts", "Extended Sobel",
                         "Kirsch", "Canny (Scratch)", "Canny (Library)"]
            edge_imgs = {k: results[k]["edge"] for k in edge_keys if k in results}
            self.analysis_panels["edge_compare_grid"].show_image_grid(
                edge_imgs,
                title="Edge Detection Comparison — All Methods",
                cmap="gray", cols=4)

            # Sharpening comparison grid
            sharp_imgs = {
                "Original (Gray)":     self.gray_img,
                "Laplacian Sharpened": results.get("Laplacian",        {}).get("edge"),
                "Unsharp Masking":     results.get("Unsharp Masking",  {}).get("edge"),
            }
            self.analysis_panels["sharp_compare_grid"].show_image_grid(
                sharp_imgs,
                title="Sharpening Comparison — Original vs Laplacian vs USM",
                cmap="gray", cols=3)

            self._runtime_dirty = False
            fastest = min(runtimes, key=runtimes.get)
            slowest = max(runtimes, key=runtimes.get)
            self.status.showMessage(
                f"✅  Runtime analysis done  |  "
                f"Fastest: {fastest} ({runtimes[fastest]:.1f} ms)  |  "
                f"Slowest: {slowest} ({runtimes[slowest]:.1f} ms)", 8000)
        except Exception as e:
            self.status.showMessage(f"[Runtime error] {e}", 5000)

    def _do_update(self):
        if self.gray_img is None:
            return
        method = self.method_combo.currentText()
        sigma  = self.sigma_slider.value() / 10.0
        thr    = self.thr_slider.value()  / 100.0

        enh_text = self.enh_combo.currentText()
        if "CLAHE" in enh_text: enh_m = "CLAHE"
        elif "Histogram" in enh_text: enh_m = "HE"
        elif "Contrast" in enh_text: enh_m = "CS"
        else: enh_m = "None"

        res = compute_pipeline(
            self.gray_img, method, sigma, thr,
            enh_method=enh_m,
            clahe_clip=self.enh_clahe_clip,
            cs_low=self.cs_low,
            cs_high=self.cs_high
        )
        self._last_result = res
        m_color = METHOD_COLORS.get(method, TEXT_COL)

        self.panels["acq_orig"].show_image(self.gray_img, "gray")
        self.panels["acq"].show_image(res["acq"], "gray")
        self.analysis_panels["acq_orig_ha"].show_hist_ogive(self.gray_img, label="Original Gray", color="#89B4FA")
        self.analysis_panels["acq_ha"].show_hist_ogive(res["acq"], label="Acquired Image", color="#FAB387")
        self.analysis_panels["acq_compare_ha"].show_comparison_hist_ogive(
            {"Original": self.gray_img, "Acquired": res["acq"]}, title="Original vs Acquired")

        self.panels["denoised"].show_image(res["denoised"],   "gray")
        self.panels["morpho_pre"].show_image(res["morpho_pre"], "gray")

        self.panels["gx"].show_image(res["gx"], "RdBu_r")
        self.panels["gy"].show_image(res["gy"], "PRGn_r")
        self.panels["magnitude"].show_image(res["magnitude"], EDGE_CMAP, colorbar=True)
        self.panels["direction"].show_image(res["direction"], "hsv", colorbar=True)

        self.panels["binary"].show_image(res["binary"], "gray")
        self.panels["morpho_post"].show_image(res["morpho_post"], "gray")
        self.panels["histogram"].show_histogram(res["magnitude"], thr, m_color)
        self.panels["profile"].show_profile(res["magnitude"], res["binary"], thr, m_color, row=self.gray_img.shape[0] // 2)
        self.analysis_panels["result_mag_ha"].show_hist_ogive(res["magnitude"], label=f"Edge Magnitude — {method}", color=m_color)

        self.stats_lbl.setText(
            f"Edge Method:  {method}\n"
            f"Enhancement:  {enh_m}\n"
            f"Runtime:      {res['elapsed']:.2f} ms\n"
            f"Edge Density: {res['density']:.4f}\n"
            f"Mean Mag:     {res['mean_mag']:.4f}\n"
            f"σ (sigma):    {sigma:.2f}\n"
            f"Threshold:    {thr:.2f}\n"
            f"Image:        {self.gray_img.shape[1]}×{self.gray_img.shape[0]} px\n"
            f"Source:       {self.image_name}"
        )
        self.stats_lbl.setStyleSheet(f"color:{m_color};")
        self.info_lbl.setText(KERNEL_INFO.get(method, ""))

        self.status.showMessage(
            f"  {method}  |  Enh={enh_m}  |  σ={sigma:.2f}  |  thr={thr:.2f}"
            f"  |  density={res['density']:.4f}  |  ⏱ {res['elapsed']:.1f} ms")

        self._do_update_enhancement()
        self._autosave_timer.start(3000)

    def _do_update_enhancement(self):
        if self._last_result is None or self.gray_img is None:
            return

        acq = self._last_result["acq"]
        enhs = compute_all_enhancements(
            acq,
            clahe_clip=self.enh_clahe_clip,
            cs_low=self.cs_low,
            cs_high=self.cs_high,
        )

        key_map = {"CS": "enh_cs", "HE": "enh_he", "CLAHE": "enh_clahe"}
        for mname, panel_key in key_map.items():
            self.panels[panel_key].show_image(enhs[mname], "gray")

        ha_map = {
            "CS":    ("enh_cs_ha",    "Contrast Stretching"),
            "HE":    ("enh_he_ha",    "Histogram Equalization"),
            "CLAHE": ("enh_clahe_ha", "CLAHE (Adaptive)"),
        }
        for mname, (ap_key, lbl) in ha_map.items():
            self.analysis_panels[ap_key].show_hist_ogive(
                enhs[mname], label=lbl, color=ENH_COLORS[mname])

        metrics = {mname: compute_enhancement_metrics(acq, enhs[mname]) for mname in enhs}
        self.analysis_panels["enh_perf"].show_performance_eval(metrics)

    def _do_autosave(self):
        self._save_panels(auto=True)

    def _save_all_now(self):
        self._save_panels(auto=False)

    def _save_panels(self, auto: bool = False):
        if self._last_result is None:
            return
        method = self.method_combo.currentText()
        sigma  = self.sigma_slider.value() / 10.0
        thr    = self.thr_slider.value()  / 100.0
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir = os.path.join(self.output_dir, ts)
        os.makedirs(subdir, exist_ok=True)

        suffix = f"{method.replace(' ','_')}_s{sigma:.1f}_t{thr:.2f}.png"
        ok = 0
        for key, panel in {**self.panels, **self.analysis_panels}.items():
            fname = f"{key}_{suffix}"
            if panel.save_to(os.path.join(subdir, fname)):
                ok += 1

        total = len(self.panels) + len(self.analysis_panels)
        label = "Auto-saved" if auto else "Saved"
        msg   = f"💾 {label} {ok}/{total} panels → {subdir}"
        self.autosave_lbl.setText(msg)
        self.autosave_lbl.setStyleSheet(f"color:{STAGE_COLORS['results']}; font-size:11px; padding-left:4px; font-family:'Segoe UI';")
        if not auto:
            self.status.showMessage(msg, 6000)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())