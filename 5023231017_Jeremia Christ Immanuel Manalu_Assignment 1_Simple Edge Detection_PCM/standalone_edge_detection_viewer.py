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
from skimage.measure import shannon_entropy as _shannon_entropy
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
}

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

    # 2. Restoration / Denoising (Using our manual mathematical Gaussian)
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

        self._load_default_image()

        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._do_update)

        self._autosave_timer = QTimer()
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._do_autosave)

        self.panels          = {}
        self.analysis_panels = {}

        self._build_ui()
        self._update()

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
        tab_style = (
            f"QTabWidget::pane {{ border:1px solid {GRID_COL}; border-radius: 10px; background: rgba(27, 40, 56, 0.45); padding: 4px; }}" 
            f"QTabBar::tab {{ background:{BG_DARK}; color:{SUBTEXT}; padding:10px 18px; margin-right: 4px; margin-bottom: 8px; border:1px solid {GRID_COL}; border-radius:8px; font-family:'Segoe UI'; font-size:12px; font-weight:bold; }}"
            f"QTabBar::tab:selected {{ background:{BG_CARD}; color:{TEXT_COL}; border:1px solid {STAGE_COLORS['preprocessing']}; }}"
            f"QTabBar::tab:hover {{ background:{BG_CARD}; color:{TEXT_COL}; }}"
        )
        tabs = QTabWidget()
        tabs.setStyleSheet(tab_style)
        tabs.addTab(self._build_tab_acquisition(),  " 1. Acquisition ")
        tabs.addTab(self._build_tab_enhancement(),  " 2. Image Enhancement ")
        tabs.addTab(self._build_tab_restoration(),  " 3. Restoration ")
        tabs.addTab(self._build_tab_gradient(),     " 4. Gradient and Detection ")
        tabs.addTab(self._build_tab_results(),      " 5. Results and Analysis ")
        
        tabs.addTab(self._build_tab_secret(),       " ✧ ")
        tabs.currentChanged.connect(self._on_tab_changed)

        return tabs

    def _build_tab_secret(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent; border: none;")
        return w

    def _on_tab_changed(self, index: int):
        if not hasattr(self, 'left_panel') or not hasattr(self, 'status') or not hasattr(self, 'bg_widget'):
            return

        if index == 5:
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
                    # Use mathematical rgb2gray
                    self.gray_img = manual_rgb2gray(rgb)
            else:
                from PIL import Image as PILImage
                pil = PILImage.open(path).convert("RGB").resize((300, 300))
                # Use mathematical rgb2gray
                self.gray_img = manual_rgb2gray(np.array(pil))
                
            self.image_name = os.path.basename(path)
            self._update()
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