#!/usr/bin/env python3
"""
standalone_nuclei_seg_gui.py: MoNuSeg 2018 Nuclei Segmentation GUI
PyQt6 + Matplotlib

Usage:
    python standalone_nuclei_seg_gui.py

Optionally place a file named  background.jpg/background.png
in the same directory to enable the custom background feature
"""

#  SECTION 1 IMPORTS AND RUNTIME SETUP
import sys, os, time, warnings, traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from copy import deepcopy

import numpy as np
import cv2
import pandas as pd
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, binary_fill_holes

from skimage.color import rgb2hed as _skimage_rgb2hed
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import threshold_multiotsu

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QTabWidget, QScrollArea, QGroupBox, QSplitter,
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QFileDialog, QProgressBar,
    QSizePolicy, QListWidget, QListWidgetItem, QStatusBar,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import (
    QColor, QFont, QPixmap, QPainter, QBrush,
    QLinearGradient, QGradient, QPalette,
)

warnings.filterwarnings("ignore")

try:
    import tifffile; _USE_TIFFFILE = True
except ImportError:
    _USE_TIFFFILE = False

GPU_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpnd
    cp.zeros(1); GPU_AVAILABLE = True
except Exception:
    pass

#  SECTION 2 COLOUR PALETTE AND STYLESHEET
PAL = {
    "bg0":        "#050D1A",
    "bg1":        "#081526",
    "bg2":        "#0C1D38",
    "bg3":        "#0F2544",
    "bg4":        "#132D52",
    "sidebar":    "#06101E",
    "border":     "#1A3A60",
    "border2":    "#0E2540",
    "accent1":    "#1565C0",
    "accent2":    "#42A5F5",
    "accent3":    "#00BCD4",
    "accent4":    "#00E5FF",
    "green":      "#00E676",
    "orange":     "#FF9800",
    "red":        "#F44336",
    "text0":      "#EDF2FB",
    "text1":      "#B0C8E8",
    "text2":      "#6B8FBD",
    "text3":      "#3A608A",
    "tab_active": "#1565C0",
    "btn_save":   "#1A237E",
    "btn_sh":     "#283593",
}

# Matplotlib background colours
MPL_STYLE = {
    "figure.facecolor": "#0C1D38",
    "axes.facecolor":   "#0F2544",
    "axes.edgecolor":   "#1A3A60",
    "axes.labelcolor":  "#B0C8E8",
    "axes.titlecolor":  "#EDF2FB",
    "text.color":       "#EDF2FB",
    "xtick.color":      "#B0C8E8",
    "ytick.color":      "#B0C8E8",
    "grid.color":       "#1A3A60",
    "grid.alpha":       0.4,
    "legend.facecolor": "#0F2544",
    "legend.edgecolor": "#1A3A60",
    "legend.labelcolor":"#EDF2FB",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
}
matplotlib.rcParams.update(MPL_STYLE)

DARK_QSS = """
/* ── Global reset ─────────────────────────────────────── */
QMainWindow { background-color: transparent; }
QWidget {
  color: %(text0)s;
  font-family: "Segoe UI","SF Pro Display","Helvetica Neue",Arial,sans-serif;
  font-size: 13px;
}
/* ── Scroll bars ──────────────────────────────────────── */
QScrollArea { border:none; background:transparent; }
QScrollBar:vertical { background:rgba(12,29,56,160); width:8px; border-radius:4px; }
QScrollBar::handle:vertical { background:%(border)s; min-height:20px; border-radius:4px; }
QScrollBar::handle:vertical:hover { background:%(accent1)s; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QScrollBar:horizontal { background:rgba(12,29,56,160); height:8px; border-radius:4px; }
QScrollBar::handle:horizontal { background:%(border)s; min-width:20px; border-radius:4px; }
QScrollBar::handle:horizontal:hover { background:%(accent1)s; }
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal { width:0; }
/* ── Group boxes (global default) ────────────────────── */
QGroupBox {
  border: 1px solid %(border)s;
  border-radius: 8px;
  margin-top: 14px;
  padding-top: 10px;
  background: rgba(12,29,56,190);
}
QGroupBox::title {
  subcontrol-origin: margin;
  subcontrol-position: top left;
  left: 12px;
  padding: 0 6px;
  color: %(accent2)s;
  font-weight: 700;
  font-size: 12px;
}
/* ── Tab widget ───────────────────────────────────────── */
QTabWidget::pane {
  background: rgba(8,21,38,210);
  border: 1px solid %(border)s;
  border-radius: 0 6px 6px 6px;
}
QTabBar::tab {
  background: rgba(15,37,68,200);
  color: %(text2)s;
  border: 1px solid %(border2)s;
  border-bottom: none;
  padding: 7px 14px;
  border-radius: 4px 4px 0 0;
  margin-right: 2px;
  font-size: 12px;
  font-weight: 500;
}
QTabBar::tab:selected {
  background: %(tab_active)s;
  color: %(text0)s;
  border-color: %(accent1)s;
  font-weight: 700;
}
QTabBar::tab:hover:!selected { background: %(bg4)s; color: %(text1)s; }
QTabWidget QTabBar::tab { padding: 5px 10px; font-size: 11px; }
QTabWidget QTabBar::tab:selected { background: %(accent1)s; }
/* ── Inputs ───────────────────────────────────────────── */
QLineEdit, QTextEdit {
  background: rgba(19,45,82,230);
  color: %(text0)s;
  border: 1px solid %(border)s;
  border-radius: 4px;
  padding: 4px 8px;
  selection-background-color: %(accent1)s;
}
QLineEdit:focus, QTextEdit:focus { border-color: %(accent2)s; }
QSpinBox, QDoubleSpinBox {
  background: rgba(19,45,82,230);
  color: %(text0)s;
  border: 1px solid %(border)s;
  border-radius: 4px;
  padding: 3px 5px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: %(accent2)s; }
QSpinBox::up-button,   QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
  width: 14px; background: %(bg3)s; border: none;
}
QComboBox {
  background: rgba(19,45,82,230);
  color: %(text0)s;
  border: 1px solid %(border)s;
  border-radius: 4px;
  padding: 4px 8px;
}
QComboBox:focus { border-color: %(accent2)s; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
  background: %(bg3)s;
  color: %(text0)s;
  border: 1px solid %(border)s;
  selection-background-color: %(accent1)s;
}
/* ── Checkbox ─────────────────────────────────────────── */
QCheckBox { color: %(text1)s; spacing: 6px; }
QCheckBox::indicator {
  width: 15px; height: 15px;
  border: 1px solid %(border)s;
  border-radius: 3px;
  background: rgba(19,45,82,200);
}
QCheckBox::indicator:checked  { background: %(accent1)s; border-color: %(accent2)s; }
QCheckBox::indicator:hover    { border-color: %(accent2)s; }
/* ── Buttons ──────────────────────────────────────────── */
QPushButton {
  background: rgba(15,37,68,220);
  color: %(text0)s;
  border: 1px solid %(border)s;
  border-radius: 5px;
  padding: 6px 14px;
  font-weight: 500;
}
QPushButton:hover   { background: %(bg4)s; border-color: %(accent2)s; }
QPushButton:pressed { background: %(accent1)s; }
QPushButton#btnRun {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #1B6F24, stop:1 #0F5217);
  color: #CCFFCC;
  border: 1px solid #2E7D32;
  font-weight: 700;
  font-size: 13px;
  padding: 10px 20px;
}
QPushButton#btnRun:hover {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #2E7D32, stop:1 #1B5E20);
}
QPushButton#btnRun:disabled { background: %(bg3)s; color: %(text3)s; border-color: %(border2)s; }
QPushButton#btnSave {
  background: %(btn_save)s;
  color: %(accent2)s;
  border: 1px solid %(accent1)s;
  font-weight: 600;
}
QPushButton#btnSave:hover { background: %(btn_sh)s; }
QPushButton#btnBrowse {
  background: rgba(15,37,68,200);
  color: %(text1)s;
  border: 1px solid %(border)s;
  padding: 4px 10px;
  font-size: 11px;
}
QPushButton#btnBrowse:hover { border-color: %(accent2)s; color: %(text0)s; }
/* ── Progress / list ──────────────────────────────────── */
QProgressBar {
  background: rgba(15,37,68,180);
  border: 1px solid %(border)s;
  border-radius: 3px;
  height: 5px;
  color: transparent;
}
QProgressBar::chunk {
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 %(accent1)s, stop:1 %(accent4)s);
  border-radius: 3px;
}
QListWidget {
  background: rgba(19,45,82,200);
  border: 1px solid %(border)s;
  border-radius: 4px;
}
QListWidget::item { padding:3px 6px; border-radius:3px; }
QListWidget::item:selected { background:%(accent1)s; }
QListWidget::item:hover { background:%(bg3)s; }
QStatusBar { background:%(bg0)s; color:%(text2)s;
  border-top:1px solid %(border2)s; font-size:11px; }
QSplitter::handle { background:%(border2)s; }
QSplitter::handle:hover { background:%(accent1)s; }
QToolTip { background:%(bg3)s; color:%(text0)s; border:1px solid %(accent2)s;
  padding:4px 8px; border-radius:4px; }
QLabel#sectionTitle { color:%(accent2)s; font-weight:700; font-size:12px;
  padding:2px 0; }
QLabel#imageTitle { color:%(accent4)s; font-weight:700; font-size:13px; }
""" % PAL

#  SECTION 3 PROCESSING FUNCTIONS  (from notebook v2.1, final versions)
def load_image(img_path: Path) -> np.ndarray:
    """Load TIFF/image → uint8 RGB (H, W, 3)."""
    if _USE_TIFFFILE:
        img = tifffile.imread(str(img_path))
    else:
        img = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    return img


def parse_xml_to_mask(xml_path: Path, image_shape: tuple) -> np.ndarray:
    """Parse MoNuSeg XML annotation → binary ground-truth mask."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    H, W = image_shape[:2]
    mask  = np.zeros((H, W), dtype=np.uint8)
    count = 0
    for region in root.iter("Region"):
        verts = region.find("Vertices")
        if verts is None:
            continue
        coords = []
        for v in verts.findall("Vertex"):
            try:
                x = int(np.clip(round(float(v.get("X", 0))), 0, W - 1))
                y = int(np.clip(round(float(v.get("Y", 0))), 0, H - 1))
                coords.append([x, y])
            except (ValueError, TypeError):
                continue
        if len(coords) >= 3:
            cv2.fillPoly(mask, [np.array(coords, np.int32)], 255)
            count += 1
    return mask


def extract_h_channel_rgb2hed(image_rgb: np.ndarray):
    hed   = _skimage_rgb2hed(image_rgb.astype(np.float32) / 255.0)
    H_raw = hed[:, :, 0]
    H_u8  = cv2.normalize(H_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return H_u8, H_raw


def extract_h_channel_manual(image_rgb: np.ndarray):
    img_float = np.clip(image_rgb.astype(np.float64) / 255.0, 1e-6, 1.0)
    OD = -np.log10(img_float)
    M  = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
    M_inv   = np.linalg.inv(M)
    od_flat = OD.reshape(-1, 3)
    stains  = od_flat @ M_inv
    h, w    = image_rgb.shape[:2]
    H_raw   = stains[:, 0].reshape(h, w)
    H_u8    = cv2.normalize(H_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return H_u8, H_raw


def estimate_stains_macenko(image_rgb: np.ndarray, percentile: float = 99,
                             min_od: float = 0.15):
    img     = np.clip(image_rgb.astype(np.float64) / 255.0, 1e-6, 1.0)
    OD      = -np.log10(img)
    od_flat = OD.reshape(-1, 3)
    mask    = od_flat.min(axis=1) > min_od
    od_tis  = od_flat[mask]
    if len(od_tis) < 300:
        return None, None
    _, _, Vt = np.linalg.svd(od_tis - od_tis.mean(0), full_matrices=False)
    T      = od_tis @ Vt[:2].T
    angles = np.arctan2(T[:, 1], T[:, 0])
    alpha  = np.percentile(angles, 100 - percentile)
    beta   = np.percentile(angles, percentile)
    vec1   = np.array([np.cos(alpha), np.sin(alpha)]) @ Vt[:2]
    vec2   = np.array([np.cos(beta),  np.sin(beta)])  @ Vt[:2]
    vec1  /= np.linalg.norm(vec1) + 1e-12
    vec2  /= np.linalg.norm(vec2) + 1e-12
    ref_H  = np.array([0.6442, 0.7166, 0.2668]); ref_H /= np.linalg.norm(ref_H)
    H_vec, E_vec = (vec1, vec2) if np.dot(vec1, ref_H) >= np.dot(vec2, ref_H) else (vec2, vec1)
    return H_vec, E_vec


def extract_h_channel_macenko(image_rgb: np.ndarray):
    H_vec, E_vec = estimate_stains_macenko(image_rgb)
    if H_vec is None:
        return extract_h_channel_manual(image_rgb)
    R_vec = np.cross(H_vec, E_vec); R_vec /= np.linalg.norm(R_vec) + 1e-12
    M     = np.stack([H_vec, E_vec, R_vec])
    M_inv = np.linalg.inv(M)
    img   = np.clip(image_rgb.astype(np.float64) / 255.0, 1e-6, 1.0)
    OD    = -np.log10(img)
    stains = (M_inv @ OD.reshape(-1, 3).T).T
    stains = np.clip(stains, 0, None)
    h, w   = image_rgb.shape[:2]
    H_raw  = stains[:, 0].reshape(h, w)
    H_u8   = cv2.normalize(H_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return H_u8, H_raw


def get_h_channel(image_rgb: np.ndarray, mode: str = "rgb2hed"):
    if mode == "macenko":   return extract_h_channel_macenko(image_rgb)
    elif mode == "manual":  return extract_h_channel_manual(image_rgb)
    else:                   return extract_h_channel_rgb2hed(image_rgb)


def adaptive_threshold(blur_img: np.ndarray, strategy: str = "otsu",
                        pct: float = 75.0) -> np.ndarray:
    if strategy == "multi_otsu":
        try:
            thresholds = threshold_multiotsu(blur_img, classes=3)
            _, binary  = cv2.threshold(blur_img, float(thresholds[-1]), 255, cv2.THRESH_BINARY)
        except ValueError:
            _, binary  = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif strategy == "percentile":
        thresh_val = np.percentile(blur_img, pct)
        _, binary  = cv2.threshold(blur_img, thresh_val, 255, cv2.THRESH_BINARY)
    elif strategy == "auto":
        _, tentative = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_frac = (tentative > 0).mean()
        if fg_frac > 0.60:
            thresh_val = np.percentile(blur_img, pct)
            _, binary  = cv2.threshold(blur_img, thresh_val, 255, cv2.THRESH_BINARY)
        else:
            _, binary  = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:   # "otsu"
        _, binary  = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def segment_nuclei(image_rgb, img_name="", params=None,
                   use_gpu=False, use_ws=False,
                   threshold_map=None, percentile_map=None,
                   stain_mode="rgb2hed"):
    """Full V2.1 nucleus segmentation pipeline — vectorized."""
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if threshold_map is None:
        threshold_map = {}
    if percentile_map is None:
        percentile_map = {}

    H_u8, _ = get_h_channel(image_rgb, mode=stain_mode)

    if params.get("use_clahe", True):
        clahe = cv2.createCLAHE(clipLimit=params["clahe_clip_limit"],
                                 tileGridSize=params["clahe_tile_size"])
        H_u8  = clahe.apply(H_u8)

    blur = cv2.GaussianBlur(H_u8, params["gaussian_ksize"], 0)

    strategy = threshold_map.get(img_name, "otsu")
    pct_val  = percentile_map.get(img_name, 68.0)
    binary   = adaptive_threshold(blur, strategy, pct=pct_val)

    ks = params["morph_kernel_size"]
    k  = np.ones((ks, ks), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=params["open_iterations"])
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=params["close_iterations"])
    binary = (binary_fill_holes(binary > 0) * 255).astype(np.uint8)

    if use_ws:
        bb = binary.astype(bool)
        if use_gpu and GPU_AVAILABLE:
            dist = cp.asnumpy(cpnd.distance_transform_edt(cp.asarray(bb.astype(np.float32))))
        else:
            dist = distance_transform_edt(bb)
        dn         = dist / (dist.max() + 1e-8)
        thresh_abs = params["dist_thresh_frac"] * dn.max()
        try:
            coords = peak_local_max(dn, min_distance=params["peak_min_dist"],
                                    threshold_abs=thresh_abs, labels=bb)
        except TypeError:
            coords = peak_local_max(dn, min_distance=params["peak_min_dist"],
                                    threshold_abs=thresh_abs)
        if len(coords):
            mkrs = np.zeros(binary.shape, np.int32)
            mkrs[coords[:, 0], coords[:, 1]] = 1
            mkrs, _ = ndi.label(mkrs)
            try:    lbls = watershed(-dn, mkrs, mask=bb, compactness=0.001)
            except: lbls = watershed(-dn, mkrs, mask=bb)
            binary = (lbls > 0).astype(np.uint8) * 255

    # Vectorised size filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    valid = np.where((areas >= params["min_area_px"]) & (areas <= params["max_area_px"]))[0]
    valid = valid[valid > 0]
    binary = np.isin(labels, valid).astype(np.uint8) * 255
    return binary


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Pixel-level IoU, Dice, Precision, Recall, TP/FP/FN/TN."""
    p  = pred.astype(bool); g = gt.astype(bool)
    tp = int(np.logical_and( p,  g).sum())
    fp = int(np.logical_and( p, ~g).sum())
    fn = int(np.logical_and(~p,  g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    return dict(
        IoU       = tp / (tp + fp + fn + 1e-8),
        Dice      = 2 * tp / (2 * tp + fp + fn + 1e-8),
        Precision = tp / (tp + fp + 1e-8),
        Recall    = tp / (tp + fn + 1e-8),
        TP=tp, FP=fp, FN=fn, TN=tn,
    )


#  DATA-COLLECTION functions
def collect_stepwise_data(image_rgb, gt_mask, img_name, params,
                           threshold_map, percentile_map, stain_mode):
    """Run pipeline stage-by-stage; return raw data for plotting."""
    stages = [
        "1. Raw H\n(Otsu base)",
        "2. + Gauss\nblur",
        "3. + CLAHE\n(if on)",
        "4. + Adaptive\nthreshold",
        "5. + Morph\nopen",
        "6. + Close\n+ fill",
        "7. Final\n(size filt.)",
    ]
    strategy = threshold_map.get(img_name, "otsu")
    pct_val  = percentile_map.get(img_name, 68.0)
    ks = params["morph_kernel_size"]
    k  = np.ones((ks, ks), np.uint8)

    masks = []
    H_u8_raw, _ = get_h_channel(image_rgb, mode=stain_mode)
    _, s0 = cv2.threshold(H_u8_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(s0)

    blur_s1 = cv2.GaussianBlur(H_u8_raw, params["gaussian_ksize"], 0)
    _, s1   = cv2.threshold(blur_s1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(s1)

    H_cl = H_u8_raw
    if params.get("use_clahe", True):
        clahe = cv2.createCLAHE(clipLimit=params["clahe_clip_limit"],
                                  tileGridSize=params["clahe_tile_size"])
        H_cl  = clahe.apply(H_u8_raw)
    blur_s2 = cv2.GaussianBlur(H_cl, params["gaussian_ksize"], 0)
    _, s2   = cv2.threshold(blur_s2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(s2)

    s3 = adaptive_threshold(blur_s2, strategy, pct=pct_val); masks.append(s3)
    s4 = cv2.morphologyEx(s3, cv2.MORPH_OPEN,  k, iterations=params["open_iterations"]); masks.append(s4)
    s5 = cv2.morphologyEx(s4, cv2.MORPH_CLOSE, k, iterations=params["close_iterations"])
    s5 = (binary_fill_holes(s5 > 0) * 255).astype(np.uint8); masks.append(s5)

    s6 = s5.copy()
    nl, labs, sts, _ = cv2.connectedComponentsWithStats(s6, connectivity=8)
    areas = sts[:, cv2.CC_STAT_AREA]
    valid = np.where((areas >= params["min_area_px"]) & (areas <= params["max_area_px"]))[0]
    valid = valid[valid > 0]
    s6    = np.isin(labs, valid).astype(np.uint8) * 255
    masks.append(s6)

    ious  = [compute_metrics(m, gt_mask)["IoU"]  for m in masks]
    dices = [compute_metrics(m, gt_mask)["Dice"] for m in masks]
    return dict(stages=stages, masks=masks, ious=ious, dices=dices)


def collect_timing_data(image_rgb, img_name, params, threshold_map,
                         percentile_map, stain_mode, n_repeats=3):
    """Time each pipeline stage; return raw data for plotting."""
    lbls = [
        "1. Stain\nextraction",
        "2. CLAHE",
        "3. Gaussian\nblur",
        "4. Adaptive\nthreshold",
        "5. Morph\nopen+close",
        "6. Hole\nfill",
        "7. CC\nlabeling",
        "8. Size-filter\nloop",
    ]
    strategy = threshold_map.get(img_name, "otsu")
    pct_val  = percentile_map.get(img_name, 68.0)
    ks = params["morph_kernel_size"]
    k  = np.ones((ks, ks), np.uint8)
    times = {l: [] for l in lbls}
    n_cc  = 0

    for _ in range(n_repeats):
        t = time.perf_counter()
        H_u8, _ = get_h_channel(image_rgb, mode=stain_mode)
        times[lbls[0]].append(time.perf_counter() - t)

        t = time.perf_counter()
        H_cl = H_u8
        if params.get("use_clahe", True):
            cl   = cv2.createCLAHE(clipLimit=params["clahe_clip_limit"],
                                    tileGridSize=params["clahe_tile_size"])
            H_cl = cl.apply(H_u8)
        times[lbls[1]].append(time.perf_counter() - t)

        t    = time.perf_counter()
        blur = cv2.GaussianBlur(H_cl, params["gaussian_ksize"], 0)
        times[lbls[2]].append(time.perf_counter() - t)

        t      = time.perf_counter()
        binary = adaptive_threshold(blur, strategy, pct=pct_val)
        times[lbls[3]].append(time.perf_counter() - t)

        t        = time.perf_counter()
        binary_m = cv2.morphologyEx(binary,   cv2.MORPH_OPEN,  k, iterations=params["open_iterations"])
        binary_m = cv2.morphologyEx(binary_m, cv2.MORPH_CLOSE, k, iterations=params["close_iterations"])
        times[lbls[4]].append(time.perf_counter() - t)

        t        = time.perf_counter()
        binary_f = (binary_fill_holes(binary_m > 0) * 255).astype(np.uint8)
        times[lbls[5]].append(time.perf_counter() - t)

        t             = time.perf_counter()
        lbl_arr, n_cc = ndi.label(binary_f > 0)
        times[lbls[6]].append(time.perf_counter() - t)

        t   = time.perf_counter()
        out = binary_f.copy()
        for i in range(1, n_cc + 1):
            pix = lbl_arr == i
            if pix.sum() < params["min_area_px"] or pix.sum() > params["max_area_px"]:
                out[pix] = 0
        times[lbls[7]].append(time.perf_counter() - t)

    vals_ms = [np.mean(times[l]) * 1000 for l in lbls]
    total   = sum(vals_ms)
    return dict(stages=lbls, times_ms=vals_ms, total_ms=total, n_components=n_cc)


#  SECTION 4 FIGURE-CREATION FUNCTIONS
def _ax_off(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def create_histogram_figure(image_rgb: np.ndarray, H_u8: np.ndarray, title: str) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"RGB & H-Channel Histogram  ·  {title}", fontsize=11, fontweight="bold",
                 color=PAL["text0"])

    axes[0, 0].imshow(image_rgb); axes[0, 0].set_title("Original H&E", color=PAL["text0"])
    axes[0, 1].imshow(H_u8, cmap="gray"); axes[0, 1].set_title("H Channel (extracted)", color=PAL["text0"])
    _ax_off(axes[0, 0]); _ax_off(axes[0, 1])

    ax = axes[1, 0]
    for ch, col, lbl in zip([0,1,2], ["#F44336","#66BB6A","#42A5F5"], ["R","G","B"]):
        hist, bins = np.histogram(image_rgb[:, :, ch].ravel(), bins=256, range=(0,256))
        ax.plot(bins[:-1], hist, color=col, alpha=0.85, linewidth=1.4, label=lbl)
    ax.set_title("RGB Channel Histograms", color=PAL["text0"])
    ax.set_xlabel("Pixel Value"); ax.set_ylabel("Count")
    ax.legend(fontsize=9); ax.set_xlim(0, 255)
    ax.tick_params(colors=PAL["text1"]); ax.set_facecolor(PAL["bg3"])

    ax2 = axes[1, 1]
    hist_h, bins_h = np.histogram(H_u8.ravel(), bins=256, range=(0,256))
    ax2.bar(bins_h[:-1], hist_h, width=1, color="#9575CD", alpha=0.85, label="H-channel")
    otsu_val, _ = cv2.threshold(H_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pct75       = np.percentile(H_u8, 75)
    ax2.axvline(otsu_val, color="#F44336", lw=2, linestyle="--",
                label=f"Otsu = {int(otsu_val)}")
    ax2.axvline(pct75, color="#FF9800", lw=2, linestyle="-.",
                label=f"75th-pct = {pct75:.0f}")
    ax2.set_title("H-Channel Histogram + Threshold Candidates", color=PAL["text0"])
    ax2.set_xlabel("Pixel Value"); ax2.set_ylabel("Count")
    ax2.legend(fontsize=9); ax2.set_xlim(0, 255)
    ax2.tick_params(colors=PAL["text1"]); ax2.set_facecolor(PAL["bg3"])

    fig.tight_layout()
    return fig


def create_clahe_figure(H_u8_raw: np.ndarray, params: dict, title: str) -> Figure:
    clahe = cv2.createCLAHE(clipLimit=params["clahe_clip_limit"],
                              tileGridSize=params["clahe_tile_size"])
    H_cl  = clahe.apply(H_u8_raw)
    diff  = H_cl.astype(np.int16) - H_u8_raw.astype(np.int16)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"CLAHE Diagnostic  ·  {title}\n"
        f"clipLimit={params['clahe_clip_limit']}  tileGridSize={params['clahe_tile_size']}",
        fontsize=11, fontweight="bold", color=PAL["text0"])

    axes[0, 0].imshow(H_u8_raw, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Before CLAHE", color=PAL["text0"]); _ax_off(axes[0, 0])
    axes[0, 1].imshow(H_cl, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("After CLAHE", color=PAL["text0"]); _ax_off(axes[0, 1])
    im = axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-80, vmax=80)
    axes[0, 2].set_title("Difference (After − Before)", color=PAL["text0"]); _ax_off(axes[0, 2])
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    bins  = np.arange(257)
    h_bef = np.histogram(H_u8_raw.ravel(), bins=bins)[0].astype(float)
    h_aft = np.histogram(H_cl.ravel(),     bins=bins)[0].astype(float)

    for ax, data, color, title_lbl in [
        (axes[1, 0], h_bef, "#42A5F5", "Histogram — Before CLAHE"),
        (axes[1, 1], h_aft, "#FF9800",  "Histogram — After CLAHE"),
    ]:
        ax.bar(bins[:-1], data, width=1, color=color, alpha=0.85)
        ax.set_title(title_lbl, color=PAL["text0"])
        ax.set_xlabel("Pixel Value"); ax.set_ylabel("Count"); ax.set_xlim(0, 255)
        ax.tick_params(colors=PAL["text1"]); ax.set_facecolor(PAL["bg3"])

    cdf_bef = np.cumsum(h_bef) / h_bef.sum()
    cdf_aft = np.cumsum(h_aft) / h_aft.sum()
    x       = bins[:-1]
    ax3 = axes[1, 2]
    ax3.plot(x, cdf_bef, color="#42A5F5", lw=2, label="Before CLAHE")
    ax3.plot(x, cdf_aft, color="#FF9800", lw=2, label="After CLAHE")
    ax3.plot([0, 255], [0, 1], color=PAL["text2"], lw=1, linestyle="--", alpha=0.6,
             label="Ideal (uniform)")
    ax3.set_title("CDF Comparison\n(closer to diagonal = more uniform)", color=PAL["text0"])
    ax3.set_xlabel("Pixel Value"); ax3.set_ylabel("CDF")
    ax3.legend(fontsize=9); ax3.set_xlim(0, 255)
    ax3.tick_params(colors=PAL["text1"]); ax3.set_facecolor(PAL["bg3"])

    dstd = H_cl.std() - H_u8_raw.std()
    fig.text(0.5, 0.01,
             f"Std: {H_u8_raw.std():.1f} → {H_cl.std():.1f}  (Δ={dstd:+.1f})   "
             f"Dynamic range: {H_cl.min()}–{H_cl.max()}",
             ha="center", fontsize=10, color=PAL["text2"])
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def create_segmentation_figure(image_rgb, gt_mask, pred_mask, metrics, title) -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"{title}\n"
        f"IoU={metrics['IoU']:.4f}  Dice={metrics['Dice']:.4f}  "
        f"Prec={metrics['Precision']:.4f}  Rec={metrics['Recall']:.4f}",
        fontsize=11, fontweight="bold", color=PAL["text0"])

    def blend(img, mask, colour, alpha=0.55):
        ov = img.astype(np.float32).copy()
        ov[mask > 0] = ov[mask > 0] * (1 - alpha) + np.array(colour) * alpha
        return np.clip(ov, 0, 255).astype(np.uint8)

    axes[0,0].imshow(image_rgb);  axes[0,0].set_title("Original H&E", color=PAL["text0"])
    axes[0,1].imshow(gt_mask,   cmap="Greens");  axes[0,1].set_title("Ground Truth", color=PAL["text0"])
    axes[0,2].imshow(pred_mask, cmap="Oranges"); axes[0,2].set_title("Predicted Mask", color=PAL["text0"])
    axes[1,0].imshow(blend(image_rgb, gt_mask,   [0,220,0]));   axes[1,0].set_title("GT Overlay", color=PAL["text0"])
    axes[1,1].imshow(blend(image_rgb, pred_mask, [255,140,0])); axes[1,1].set_title("Prediction Overlay", color=PAL["text0"])

    g = gt_mask > 0; p = pred_mask > 0
    em = np.zeros((*gt_mask.shape, 3), np.uint8)
    em[np.logical_and( p,  g)] = [0,  210,  0]
    em[np.logical_and( p, ~g)] = [220,  0,  0]
    em[np.logical_and(~p,  g)] = [0,    0, 220]
    axes[1,2].imshow(em)
    axes[1,2].set_title("Error Map  (TP / FP / FN)", color=PAL["text0"])
    axes[1,2].legend(
        handles=[mpatches.Patch(color="#00D200", label=f"TP {metrics['TP']:,}"),
                 mpatches.Patch(color="#DC0000", label=f"FP {metrics['FP']:,}"),
                 mpatches.Patch(color="#0000DC", label=f"FN {metrics['FN']:,}")],
        loc="lower right", fontsize=9)

    for ax in axes.flat:
        _ax_off(ax)
    fig.tight_layout()
    return fig


def create_stepwise_bar_figure(stages, ious, dices, title) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Step-by-Step Performance  ·  {title}", fontsize=11, fontweight="bold",
                 color=PAL["text0"])

    x     = np.arange(len(stages))
    width = 0.38
    ci    = ["#5C9BD4"] * (len(x) - 1) + ["#1565C0"]
    cd    = ["#FFA040"] * (len(x) - 1) + ["#E65100"]

    ax = axes[0]
    bi = ax.bar(x - width/2, ious,  width, color=ci, edgecolor="#1A3A60", lw=0.7, label="IoU")
    bd = ax.bar(x + width/2, dices, width, color=cd, edgecolor="#1A3A60", lw=0.7, label="Dice")
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=8, color=PAL["text1"])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.set_title("Absolute Score per Stage", color=PAL["text0"])
    ax.legend(fontsize=9); ax.tick_params(colors=PAL["text1"]); ax.set_facecolor(PAL["bg3"])
    for bar, val in zip(list(bi) + list(bd), ious + dices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=7.5, fontweight="bold", color=PAL["text0"])

    ax2 = axes[1]
    di = [v - ious[0]  for v in ious]
    dd = [v - dices[0] for v in dices]
    ci2 = ["#43A047" if d >= 0 else "#E53935" for d in di]
    cd2 = ["#FB8C00" if d >= 0 else "#8E24AA" for d in dd]
    ax2.bar(x - width/2, di, width, color=ci2, edgecolor="#1A3A60", lw=0.7, label="ΔIoU")
    ax2.bar(x + width/2, dd, width, color=cd2, edgecolor="#1A3A60", lw=0.7, label="ΔDice")
    ax2.axhline(0, color=PAL["text2"], lw=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(stages, fontsize=8, color=PAL["text1"])
    ax2.set_ylabel("Δ vs Stage ①")
    ax2.set_title("Incremental Contribution", color=PAL["text0"])
    ax2.legend(fontsize=9); ax2.tick_params(colors=PAL["text1"]); ax2.set_facecolor(PAL["bg3"])

    fig.tight_layout()
    return fig


def create_stepwise_grid_figure(image_rgb, gt_mask, masks, stages, ious, dices, title) -> Figure:
    n_cols = len(stages)
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 3.0, 9))
    fig.suptitle(f"Intermediate Mask Grid  ·  {title}", fontsize=11, fontweight="bold",
                 color=PAL["text0"])
    gt_b = gt_mask > 0

    for j, (sname, mask_s, iou_s, dice_s) in enumerate(zip(stages, masks, ious, dices)):
        axes[0, j].imshow(mask_s, cmap="gray")
        axes[0, j].set_title(sname, fontsize=7.5, color=PAL["text0"])

        pb  = mask_s > 0
        err = np.zeros((*mask_s.shape, 3), np.uint8)
        err[np.logical_and( pb,  gt_b)] = [0,  200,  0]
        err[np.logical_and( pb, ~gt_b)] = [220,  0,  0]
        err[np.logical_and(~pb,  gt_b)] = [0,    0, 220]
        axes[1, j].imshow(err)
        axes[1, j].set_title(f"IoU={iou_s:.3f}\nDice={dice_s:.3f}", fontsize=7.5, color=PAL["text0"])

        ov = image_rgb.copy().astype(np.float32)
        ov[mask_s > 0] = ov[mask_s > 0] * 0.45 + np.array([255, 165, 0]) * 0.55
        axes[2, j].imshow(ov.astype(np.uint8))
        axes[2, j].set_title("Overlay", fontsize=7, color=PAL["text0"])

    for ax in axes.flat:
        _ax_off(ax)
    fig.tight_layout()
    return fig


def create_timing_figure(stages, times_ms, total_ms, n_cc, title) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    clrs = ["#42A5F5"] * len(stages)
    clrs[-1] = "#FF6F00"   # highlight size-filter as the bottleneck
    bars = ax.barh(stages, times_ms, color=clrs, edgecolor=PAL["border"], lw=0.7)
    ax.set_xlabel(f"Time (ms)", color=PAL["text1"])
    ax.set_title(
        f"Per-Stage Timing  ·  {title}\n"
        f"Total ≈ {total_ms:.1f} ms   |   CC before size-filter: {n_cc}",
        fontsize=11, color=PAL["text0"])
    ax.set_facecolor(PAL["bg3"])
    ax.tick_params(colors=PAL["text1"])
    ax.invert_yaxis()
    if times_ms:
        mx = max(times_ms)
        for bar, v in zip(bars, times_ms):
            ax.text(bar.get_width() + mx * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f} ms ({v/total_ms*100:.0f}%)" if total_ms > 0 else f"{v:.1f}",
                    va="center", fontsize=8.5, color=PAL["text0"])
    fig.tight_layout()
    return fig


def create_summary_figure(df: pd.DataFrame) -> Figure:
    metrics = ["IoU", "Dice", "Precision", "Recall"]
    n = len(df)
    short = [name.split("-")[1] + "\n" + name.split("-")[2][:5] for name in df["Image"]]
    colours = ["#42A5F5", "#66BB6A", "#FF9800", "#E91E63"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Segmentation Performance Summary — All Images",
                 fontsize=12, fontweight="bold", color=PAL["text0"])

    for ax, met in zip(axes.flat, metrics):
        vals = df[met].values
        bars = ax.bar(short, vals, color=colours, edgecolor=PAL["border"], lw=0.7)
        ax.set_title(met, fontsize=11, color=PAL["text0"])
        ax.set_ylim(0, min(max(vals) * 1.3, 1.1))
        mean_v = vals.mean()
        ax.axhline(mean_v, color="#F44336", ls="--", lw=1.5, label=f"Mean={mean_v:.3f}")
        ax.legend(fontsize=9); ax.set_facecolor(PAL["bg3"])
        ax.tick_params(colors=PAL["text1"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f"{v:.4f}", ha="center", fontsize=9, fontweight="bold", color=PAL["text0"])

    fig.tight_layout()
    return fig


def create_runtime_figure(df: pd.DataFrame) -> Figure:
    short = [name.split("-")[1] + "\n" + name.split("-")[2][:5] for name in df["Image"]]
    colours = ["#42A5F5", "#66BB6A", "#FF9800", "#E91E63"]
    vals = df["Running Time"].values
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Running Time per Image", fontsize=11, fontweight="bold", color=PAL["text0"])
    bars = ax.bar(short, vals, color=colours, edgecolor=PAL["border"], lw=0.7)
    ax.set_ylabel("Time (s)"); ax.set_ylim(0, max(vals) * 1.3)
    ax.axhline(vals.mean(), color="#F44336", ls="--", lw=1.5, label=f"Mean={vals.mean():.3f}s")
    ax.legend(fontsize=9); ax.set_facecolor(PAL["bg3"]); ax.tick_params(colors=PAL["text1"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f"{v:.3f}s", ha="center", fontsize=9, fontweight="bold", color=PAL["text0"])
    fig.tight_layout()
    return fig


def create_cross_step_figure(step_results: dict) -> Figure:
    if not step_results:
        return Figure()
    stages = list(step_results.values())[0]["stages"]
    n_s    = len(stages)
    x      = np.arange(n_s)
    width  = 0.20
    pal4   = ["#42A5F5", "#66BB6A", "#FF9800", "#E91E63"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Step-by-Step Performance — All Images", fontsize=12, fontweight="bold",
                 color=PAL["text0"])

    for idx, (name, sr) in enumerate(step_results.items()):
        sn  = name.split("-")[1]
        off = (idx - 1.5) * width
        axes[0].bar(x + off, sr["ious"],  width, color=pal4[idx], alpha=0.85,
                    edgecolor=PAL["border"], lw=0.5, label=sn)
        axes[1].bar(x + off, sr["dices"], width, color=pal4[idx], alpha=0.85,
                    edgecolor=PAL["border"], lw=0.5, label=sn)

    for ax, metric in zip(axes, ["IoU per Stage", "Dice per Stage"]):
        ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=7.5, color=PAL["text1"])
        ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_title(metric, color=PAL["text0"])
        ax.legend(fontsize=8, title="Image", title_fontsize=8)
        ax.tick_params(colors=PAL["text1"]); ax.set_facecolor(PAL["bg3"])
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def create_cross_timing_figure(timing_results: dict) -> Figure:
    if not timing_results:
        return Figure()
    stage_lbls = list(timing_results.values())[0]["stages"]
    xt    = np.arange(len(stage_lbls))
    wt    = 0.20
    pal4  = ["#42A5F5", "#66BB6A", "#FF9800", "#E91E63"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Per-Stage Timing — All Images", fontsize=12, fontweight="bold", color=PAL["text0"])
    for idx, (name, tr) in enumerate(timing_results.items()):
        sn  = name.split("-")[1]
        off = (idx - 1.5) * wt
        ax.bar(xt + off, tr["times_ms"], wt, color=pal4[idx], alpha=0.85,
               edgecolor=PAL["border"], lw=0.5, label=sn)
    ax.set_xticks(xt); ax.set_xticklabels(stage_lbls, fontsize=8, color=PAL["text1"])
    ax.set_ylabel("Time (ms)"); ax.set_title("Per-Stage Time (ms, mean of 3 runs)", color=PAL["text0"])
    ax.legend(fontsize=9, title="Image"); ax.tick_params(colors=PAL["text1"])
    ax.set_facecolor(PAL["bg3"]); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


#  SECTION 5 DEFAULT CONFIG
DEFAULT_IMAGE_NAMES = [
    "TCGA-AR-A1AS-01Z-00-DX1",
    "TCGA-AY-A8YK-01A-01-TS1",
    "TCGA-E2-A1B5-01Z-00-DX1",
    "TCGA-RD-A8N9-01A-01-TS1",
]
IMG_SHORT = {
    "TCGA-AR-A1AS-01Z-00-DX1": "AR-A1AS",
    "TCGA-AY-A8YK-01A-01-TS1": "AY-A8YK",
    "TCGA-E2-A1B5-01Z-00-DX1": "E2-A1B5",
    "TCGA-RD-A8N9-01A-01-TS1": "RD-A8N9",
}
DEFAULT_THRESHOLD_MAP = {n: "percentile" for n in DEFAULT_IMAGE_NAMES}
DEFAULT_PERCENTILE_MAP = {
    "TCGA-AR-A1AS-01Z-00-DX1": 66.4,
    "TCGA-AY-A8YK-01A-01-TS1": 68.0,
    "TCGA-E2-A1B5-01Z-00-DX1": 82.8,
    "TCGA-RD-A8N9-01A-01-TS1": 61.3,
}
DEFAULT_PARAMS = {
    "use_clahe":        True,
    "clahe_clip_limit": 1.0,
    "clahe_tile_size":  (9, 9),
    "gaussian_ksize":   (5, 5),
    "morph_kernel_size":3,
    "open_iterations":  2,
    "close_iterations": 0,
    "min_area_px":      20,
    "max_area_px":      80000,
    "peak_min_dist":    8,
    "dist_thresh_frac": 0.28,
}

#  SECTION 6 WORKER THREAD
class ProcessingWorker(QThread):
    log        = pyqtSignal(str)
    progress   = pyqtSignal(int, int, str)    # current, total, status

    hist_ready    = pyqtSignal(str, object, object)          # name, rgb, H_u8
    clahe_ready   = pyqtSignal(str, object, dict)            # name, H_raw, params
    seg_ready     = pyqtSignal(str, object, object, object, dict)  # name, rgb, gt, pred, metrics
    step_ready    = pyqtSignal(str, dict)                    # name, step_data
    timing_ready  = pyqtSignal(str, dict)                    # name, timing_data

    all_done   = pyqtSignal(list)
    error      = pyqtSignal(str)
    finished_  = pyqtSignal()

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        cfg      = self.cfg
        names    = cfg["image_names"]
        base_dir = Path(cfg["base_dir"])
        results  = []

        for i, name in enumerate(names):
            if self._abort:
                self.log.emit("⚠  Aborted by user.")
                break
            self.progress.emit(i, len(names), name)
            self.log.emit(f"\n{'═'*56}\n  Processing: {IMG_SHORT.get(name, name)}\n{'═'*56}")

            img_path = base_dir / "Tissue Images" / f"{name}.tif"
            xml_path = base_dir / "Annotations"   / f"{name}.xml"

            try:
                image = load_image(img_path)
                self.log.emit(f"  ✓ Image loaded  {image.shape}  {image.dtype}")
            except Exception as e:
                self.log.emit(f"  ✗ Cannot load image: {e}")
                self.error.emit(f"Cannot load image for {name}: {e}")
                continue

            try:
                gt_mask = parse_xml_to_mask(xml_path, image.shape)
                n_nuc   = ndi.label(gt_mask > 0)[1]
                self.log.emit(f"  ✓ Ground truth parsed  ({n_nuc} nuclei)")
            except Exception as e:
                self.log.emit(f"  ✗ Cannot parse XML: {e}")
                self.error.emit(f"Cannot parse annotation for {name}: {e}")
                continue

            # Histogram 
            try:
                H_u8_raw, _ = get_h_channel(image, mode=cfg["stain_mode"])
                self.hist_ready.emit(name, image.copy(), H_u8_raw.copy())
                self.log.emit("  ✓ Histogram data ready")
            except Exception as e:
                self.log.emit(f"  ✗ Histogram error: {e}")

            # CLAHE diagnostic 
            try:
                self.clahe_ready.emit(name, H_u8_raw.copy(), deepcopy(cfg["params"]))
                self.log.emit("  ✓ CLAHE diagnostic data ready")
            except Exception as e:
                self.log.emit(f"  ✗ CLAHE error: {e}")

            # Step-wise diagnostic 
            if cfg.get("run_diagnostics", True):
                try:
                    step_data = collect_stepwise_data(
                        image, gt_mask, name, cfg["params"],
                        cfg["threshold_map"], cfg["percentile_map"], cfg["stain_mode"])
                    self.step_ready.emit(name, step_data)
                    self.log.emit("  ✓ Step-wise analysis complete")
                except Exception as e:
                    self.log.emit(f"  ✗ Step analysis error: {e}\n{traceback.format_exc()}")

            # Main segmentation
            try:
                t0 = time.perf_counter()
                pred_mask = segment_nuclei(
                    image, name, cfg["params"], False,
                    cfg["use_watershed"],
                    cfg["threshold_map"], cfg["percentile_map"], cfg["stain_mode"])
                elapsed = time.perf_counter() - t0
                metrics = compute_metrics(pred_mask, gt_mask)
                self.log.emit(
                    f"  ✓ Segmentation done in {elapsed:.3f}s\n"
                    f"     IoU={metrics['IoU']:.4f}  Dice={metrics['Dice']:.4f}")
                self.seg_ready.emit(name, image.copy(), gt_mask.copy(), pred_mask.copy(),
                                    dict(metrics))
            except Exception as e:
                elapsed = 0.0; metrics = {"IoU":0,"Dice":0,"Precision":0,"Recall":0,"TP":0,"FP":0,"FN":0,"TN":0}
                self.log.emit(f"  ✗ Segmentation error: {e}\n{traceback.format_exc()}")
                self.error.emit(f"Segmentation failed for {name}: {e}")
                continue

            # Timing diagnostic
            try:
                timing_data = collect_timing_data(
                    image, name, cfg["params"],
                    cfg["threshold_map"], cfg["percentile_map"],
                    cfg["stain_mode"], n_repeats=cfg.get("timing_repeats", 3))
                self.timing_ready.emit(name, timing_data)
                self.log.emit(f"  ✓ Timing profiled  total≈{timing_data['total_ms']:.1f}ms")
            except Exception as e:
                self.log.emit(f"  ✗ Timing error: {e}")
                timing_data = {}

            results.append({
                "Image":        name,
                "IoU":          round(metrics["IoU"],       4),
                "Dice":         round(metrics["Dice"],      4),
                "Precision":    round(metrics["Precision"], 4),
                "Recall":       round(metrics["Recall"],    4),
                "Running Time": round(elapsed,              4),
                "Threshold":    cfg["threshold_map"].get(name, "otsu"),
                "Pct":          cfg["percentile_map"].get(name, 0.0),
                "Device":       "GPU" if GPU_AVAILABLE else "CPU",
            })

        self.progress.emit(len(names), len(names), "Complete")
        self.all_done.emit(results)
        self.finished_.emit()
        self.log.emit("\n✅  All images processed.")


#  SECTION 7 CUSTOM WIDGETS
class MatplotlibCanvas(QWidget):
    """Matplotlib figure + NavigationToolbar inside a scroll area."""

    def __init__(self, parent=None, figsize=(10, 6)):
        super().__init__(parent)
        self._fig  = Figure(figsize=figsize, facecolor=PAL["bg2"])
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)

        # Style toolbar
        self._toolbar.setFixedHeight(32)
        self._toolbar.setStyleSheet(
            f"background:{PAL['bg2']}; border:none; border-bottom:1px solid {PAL['border2']};")

        # Scroll area so large figures are scrollable
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._canvas)
        self._scroll.setStyleSheet("background:transparent; border:none;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._scroll)

    def update_figure(self, fig: Figure):
        """Replace the current figure with a new one.

        Only the toolbar (index 0) is swapped out; the QScrollArea (index 1)
        is kept alive and its inner widget is replaced.  This prevents the
        'wrapped C/C++ object of type QScrollArea has been deleted' crash that
        happened when the old code called deleteLater() on every layout item
        (including self._scroll) and then tried to use self._scroll again.
        """
        plt.close(self._fig)
        self._fig = fig

        self._canvas = FigureCanvas(fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)

        new_toolbar = NavigationToolbar(self._canvas, self)
        new_toolbar.setFixedHeight(32)
        new_toolbar.setStyleSheet(
            f"background:{PAL['bg2']}; border:none;"
            f"border-bottom:1px solid {PAL['border2']};")

        layout = self.layout()
        old_item = layout.takeAt(0)          # removes old toolbar only
        if old_item and old_item.widget():
            old_item.widget().deleteLater()  # only the toolbar is deleted
        layout.insertWidget(0, new_toolbar)  # re-insert new toolbar at index 0
        self._toolbar = new_toolbar

        self._scroll.setWidget(self._canvas)

        w = int(fig.get_figwidth()  * fig.dpi)
        h = int(fig.get_figheight() * fig.dpi)
        self._canvas.setMinimumSize(w, h)
        self._canvas.draw()

    def show_placeholder(self, text="Run processing to view results"):
        """Show a placeholder message inside the canvas area."""
        self._fig.clf()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(PAL["bg3"])
        ax.text(0.5, 0.5, text, ha="center", va="center",
                fontsize=13, color=PAL["text2"],
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(PAL["border2"])
        self._canvas.draw()

#  BACKGROUND CENTRAL WIDGET
class BgCentralWidget(QWidget):
    """
    Central widget whose paintEvent draws the background image or a dark gradient
    Child containers with WA_TranslucentBackground set + rgba() QSS backgrounds
    will composite correctly, producing a glass / frosted-glass effect where the
    background image shows through the UI panels at the configured opacity level
    """
    _OVERLAY_ALPHA = 155   # 0-255 → lower = more visible background image

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg_pixmap = None

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

    def set_bg_pixmap(self, px):
        self._bg_pixmap = px
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        if self._bg_pixmap and not self._bg_pixmap.isNull():
            scaled = self._bg_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (scaled.width()  - w) // 2
            y = (scaled.height() - h) // 2
            p.drawPixmap(0, 0, scaled, x, y, w, h)

            p.fillRect(0, 0, w, h, QColor(5, 13, 26, self._OVERLAY_ALPHA))
        else:
            grad = QLinearGradient(0, 0, w, h)
            grad.setColorAt(0.0, QColor(5,  13, 26))
            grad.setColorAt(0.5, QColor(8,  21, 38))
            grad.setColorAt(1.0, QColor(12, 29, 56))
            p.fillRect(0, 0, w, h, QBrush(grad))
        p.end()


class SidebarSection(QGroupBox):
    """
    Parameter section in the sidebar
    Uses a visible accent-coloured left border and a semi-transparent
    background so the window image shows through
    """

    _SECTION_QSS = (
        "QGroupBox {"
        "  border: 1px solid #2A5C9A;"
        "  border-left: 3px solid " + PAL["accent2"] + ";"
        "  border-radius: 6px;"
        "  margin-top: 14px;"
        "  padding-top: 10px;"
        "  background: rgba(8, 21, 50, 190);"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  subcontrol-position: top left;"
        "  left: 14px;"
        "  padding: 0 6px;"
        "  color: " + PAL["accent2"] + ";"
        "  font-weight: 700;"
        "  font-size: 12px;"
        "}"
    )

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(self._SECTION_QSS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._body   = QWidget()
        self._body.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._layout = QVBoxLayout(self._body)
        self._layout.setContentsMargins(8, 4, 8, 8)
        self._layout.setSpacing(5)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._body)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def body_layout(self):
        return self._layout

    def add_row(self, label: str, widget: QWidget):
        row = QWidget()
        row.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        hl  = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)
        lbl = QLabel(label)
        lbl.setFixedWidth(120)
        lbl.setStyleSheet(f"color:{PAL['text1']}; font-size:12px;")
        hl.addWidget(lbl)
        hl.addWidget(widget)
        self._layout.addWidget(row)

    def add_widget(self, widget: QWidget):
        self._layout.addWidget(widget)

    def add_spacing(self, px: int = 4):
        self._layout.addSpacing(px)


class LogWidget(QTextEdit):
    """Auto-scrolling log panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(
            f"background:{PAL['bg0']}; color:{PAL['text2']};"
            f"border:1px solid {PAL['border2']}; font-family:monospace; font-size:11px;")

    def append_log(self, text: str):
        self.append(text)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


class MetricsBadge(QWidget):
    """Compact coloured badge showing IoU/Dice/Prec/Rec for one image"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        self._labels = {}
        for key, colour in [("IoU","#42A5F5"),("Dice","#66BB6A"),
                             ("Prec","#FF9800"),("Rec","#E91E63")]:
            lbl = QLabel(f"{key}: —")
            lbl.setStyleSheet(
                f"color:{colour}; font-weight:700; font-size:12px;"
                f"background:{PAL['bg3']}; padding:2px 8px; border-radius:4px;")
            hl.addWidget(lbl)
            self._labels[key] = lbl
        hl.addStretch()

    def update_metrics(self, metrics: dict):
        mapping = {"IoU":"IoU","Dice":"Dice","Precision":"Prec","Recall":"Rec"}
        for k, short in mapping.items():
            if k in metrics:
                self._labels[short].setText(f"{short}: {metrics[k]:.4f}")

#  SECTION 8 MAIN WINDOW
class NucleiSegApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬  MoNuSeg Nuclei Segmentation  ·  v2.1")
        self.setMinimumSize(1500, 900)
        self.resize(1720, 980)

        self._worker   = None
        self._results  = []
        self._step_all  = {}
        self._timing_all = {}
        self._canvases  = {}   # key → MatplotlibCanvas
        self._badges    = {}   # image_name → MetricsBadge

        self._central = BgCentralWidget()
        self.setCentralWidget(self._central)
        # Enable DWM compositing on Windows so child rgba backgrounds work
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        main_hl = QHBoxLayout(self._central)
        main_hl.setContentsMargins(0, 0, 0, 0)
        main_hl.setSpacing(0)

        # Sidebar
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(330)
        # Semi-transparent sidebar: rgba with border-right separator
        sidebar_scroll.setStyleSheet(
            "QScrollArea {"
            "  background: rgba(4, 10, 22, 210);"
            "  border: none;"
            f"  border-right: 1px solid {PAL['border']};"
            "}")
        sidebar_scroll.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        sidebar_inner = QWidget()
        # Inner widget is fully transparent, sections paint their own bg
        sidebar_inner.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        sidebar_inner.setStyleSheet("background: transparent;")
        sidebar_scroll.setWidget(sidebar_inner)
        self._sidebar_layout = QVBoxLayout(sidebar_inner)
        self._sidebar_layout.setContentsMargins(8, 8, 8, 12)
        self._sidebar_layout.setSpacing(10)

        self._build_sidebar()
        main_hl.addWidget(sidebar_scroll)

        # Tab area
        self._tabs = QTabWidget()
        self._tabs.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._tabs.setStyleSheet(
            "QTabWidget { margin: 0; background: transparent; }"
            "QTabWidget::pane {"
            "  background: rgba(5, 13, 30, 200);"
            f"  border: 1px solid {PAL['border']};"
            "  border-radius: 0 6px 6px 6px;"
            "}")
        self._build_main_tabs()
        main_hl.addWidget(self._tabs, 1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._prog = QProgressBar()
        self._prog.setFixedWidth(180)
        self._prog.setVisible(False)
        self._status.addPermanentWidget(self._prog)
        self._set_status("Ready — configure dataset path and click ▶ Run Processing")

        self._try_load_bg()

    #  SIDEBAR BUILD
    def _build_sidebar(self):
        sl = self._sidebar_layout

        # Logo / title
        logo = QLabel("🔬  NucleiSeg GUI")
        logo.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        logo.setStyleSheet(
            f"color:{PAL['accent4']}; font-size:16px; font-weight:800;"
            f"background: rgba(5, 15, 35, 160);"
            f"padding:10px 4px 6px 4px; border-bottom:2px solid {PAL['accent1']};"
            f"border-radius: 4px;")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(logo)

        sub = QLabel("MoNuSeg 2018  ·  Pipeline v2.1")
        sub.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        sub.setStyleSheet(
            f"color:{PAL['text2']}; font-size:11px; padding:4px 0 6px 0;"
            f"background: transparent;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(sub)

        # Dataset
        sec_ds = SidebarSection("📁  Dataset")
        self._le_base = QLineEdit("MoNuSeg2018")
        self._le_base.setPlaceholderText("Path to MoNuSeg2018 folder…")
        btn_browse = QPushButton("Browse"); btn_browse.setObjectName("btnBrowse")
        btn_browse.setFixedWidth(68)
        btn_browse.clicked.connect(self._browse_base_dir)
        row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0,0,0,0); rl.setSpacing(4)
        rl.addWidget(self._le_base, 1); rl.addWidget(btn_browse)
        sec_ds.add_widget(row)

        # Background image selector
        self._le_bg = QLineEdit("background.jpg")
        self._le_bg.setPlaceholderText("background.jpg / .png")
        btn_bg = QPushButton("Browse"); btn_bg.setObjectName("btnBrowse")
        btn_bg.setFixedWidth(68)
        btn_bg.clicked.connect(self._browse_bg_image)
        row2 = QWidget(); rl2 = QHBoxLayout(row2); rl2.setContentsMargins(0,0,0,0); rl2.setSpacing(4)
        rl2.addWidget(self._le_bg, 1); rl2.addWidget(btn_bg)
        lbl_bg = QLabel("Background image:")
        lbl_bg.setStyleSheet(f"color:{PAL['text2']}; font-size:11px;")
        sec_ds.add_widget(lbl_bg)
        sec_ds.add_widget(row2)
        sl.addWidget(sec_ds)

        # Stain extraction
        sec_st = SidebarSection("🧪  Stain Extraction")
        self._cb_stain = QComboBox()
        self._cb_stain.addItems(["rgb2hed", "macenko", "manual"])
        self._cb_stain.setToolTip(
            "rgb2hed: uses skimage (Ruifrok-Johnston matrix)\n"
            "macenko: SVD-based adaptive per-image estimation\n"
            "manual:  explicit R-J matrix in NumPy")
        sec_st.add_row("Mode:", self._cb_stain)
        sl.addWidget(sec_st)

        # Per-image threshold
        sec_thr = SidebarSection("🎯  Threshold Strategy")
        # Build sub-tabs for each image
        self._thr_tabs   = QTabWidget()
        self._thr_cb     = {}   # name → QComboBox (strategy)
        self._thr_pct    = {}   # name → QDoubleSpinBox (percentile)
        for name in DEFAULT_IMAGE_NAMES:
            short = IMG_SHORT[name]
            tab_w = QWidget()
            tl    = QFormLayout(tab_w)
            tl.setContentsMargins(4, 6, 4, 6)
            tl.setSpacing(6)

            cb = QComboBox()
            cb.addItems(["percentile", "otsu", "multi_otsu", "auto"])
            cb.setCurrentText(DEFAULT_THRESHOLD_MAP.get(name, "percentile"))
            self._thr_cb[name] = cb
            tl.addRow("Strategy:", cb)

            sp = QDoubleSpinBox()
            sp.setRange(1.0, 99.9); sp.setSingleStep(0.5); sp.setDecimals(1)
            sp.setValue(DEFAULT_PERCENTILE_MAP.get(name, 68.0))
            sp.setToolTip("Percentile value (used when strategy='percentile' or 'auto')")
            self._thr_pct[name] = sp
            tl.addRow("Percentile:", sp)

            # Show/hide percentile box based on strategy
            def _make_toggle(sp_=sp, cb_=cb):
                def toggle(text):
                    sp_.setEnabled(text in ("percentile", "auto"))
                return toggle
            cb.currentTextChanged.connect(_make_toggle())
            cb.currentTextChanged.emit(cb.currentText())

            self._thr_tabs.addTab(tab_w, short)

        self._thr_tabs.setStyleSheet(
            f"QTabBar::tab{{padding:3px 8px;font-size:10px;}}"
            f"QTabWidget::pane{{border:none;}}")
        self._thr_tabs.setFixedHeight(90)
        sec_thr.add_widget(self._thr_tabs)
        sl.addWidget(sec_thr)

        # CLAHE
        sec_cl = SidebarSection("📡  CLAHE Enhancement")
        self._ck_clahe = QCheckBox("Enable CLAHE")
        self._ck_clahe.setChecked(True)
        sec_cl.add_widget(self._ck_clahe)
        self._sp_clip = QDoubleSpinBox()
        self._sp_clip.setRange(0.1, 20.0); self._sp_clip.setSingleStep(0.1)
        self._sp_clip.setDecimals(1); self._sp_clip.setValue(1.0)
        sec_cl.add_row("Clip limit:", self._sp_clip)
        tile_w = QWidget(); tile_hl = QHBoxLayout(tile_w)
        tile_hl.setContentsMargins(0,0,0,0); tile_hl.setSpacing(4)
        self._sp_tile_w = QSpinBox(); self._sp_tile_w.setRange(2, 64); self._sp_tile_w.setValue(9)
        self._sp_tile_h = QSpinBox(); self._sp_tile_h.setRange(2, 64); self._sp_tile_h.setValue(9)
        tile_hl.addWidget(self._sp_tile_w); tile_hl.addWidget(QLabel("×")); tile_hl.addWidget(self._sp_tile_h)
        sec_cl.add_row("Tile size:", tile_w)
        sl.addWidget(sec_cl)

        # Gaussian blur
        sec_gb = SidebarSection("🌀  Gaussian Blur")
        self._sp_gkern = QSpinBox()
        self._sp_gkern.setRange(1, 31); self._sp_gkern.setValue(5)
        self._sp_gkern.setSingleStep(2)
        self._sp_gkern.setToolTip("Kernel size (must be odd)")
        sec_gb.add_row("Kernel size:", self._sp_gkern)
        sl.addWidget(sec_gb)

        # Morphology
        sec_mo = SidebarSection("🔧  Morphology")
        self._sp_mks   = QSpinBox(); self._sp_mks.setRange(1, 15); self._sp_mks.setValue(3)
        self._sp_oi    = QSpinBox(); self._sp_oi.setRange(0, 10);  self._sp_oi.setValue(2)
        self._sp_ci    = QSpinBox(); self._sp_ci.setRange(0, 10);  self._sp_ci.setValue(0)
        sec_mo.add_row("Kernel size:", self._sp_mks)
        sec_mo.add_row("Open iters:", self._sp_oi)
        sec_mo.add_row("Close iters:", self._sp_ci)
        sl.addWidget(sec_mo)

        # Size filter
        sec_sz = SidebarSection("🔍  Size Filter")
        self._sp_minA = QSpinBox();  self._sp_minA.setRange(1, 5000); self._sp_minA.setValue(20)
        self._sp_maxA = QSpinBox();  self._sp_maxA.setRange(100, 500000); self._sp_maxA.setValue(80000)
        sec_sz.add_row("Min area (px):", self._sp_minA)
        sec_sz.add_row("Max area (px):", self._sp_maxA)
        sl.addWidget(sec_sz)

        # Watershed
        sec_ws = SidebarSection("💧  Watershed")
        self._ck_ws = QCheckBox("Enable Watershed")
        self._ck_ws.setChecked(False)
        sec_ws.add_widget(self._ck_ws)
        self._sp_pmd = QSpinBox();  self._sp_pmd.setRange(1, 50); self._sp_pmd.setValue(8)
        self._sp_dtf = QDoubleSpinBox(); self._sp_dtf.setRange(0.05, 0.95)
        self._sp_dtf.setSingleStep(0.01); self._sp_dtf.setDecimals(2); self._sp_dtf.setValue(0.28)
        sec_ws.add_row("Peak min dist:", self._sp_pmd)
        sec_ws.add_row("Dist thresh:", self._sp_dtf)
        sl.addWidget(sec_ws)

        # Diagnostics
        sec_diag = SidebarSection("📊  Diagnostics")
        self._ck_diag = QCheckBox("Run step-wise diagnostics")
        self._ck_diag.setChecked(True)
        sec_diag.add_widget(self._ck_diag)
        self._sp_trep = QSpinBox(); self._sp_trep.setRange(1, 10); self._sp_trep.setValue(3)
        sec_diag.add_row("Timing repeats:", self._sp_trep)
        sl.addWidget(sec_diag)

        # Action buttons 
        sl.addSpacing(8)
        btn_run = QPushButton("▶  Run Processing")
        btn_run.setObjectName("btnRun")
        btn_run.clicked.connect(self._on_run)
        self._btn_run = btn_run
        sl.addWidget(btn_run)

        btn_save = QPushButton("💾  Save Results CSV")
        btn_save.setObjectName("btnSave")
        btn_save.setEnabled(False)
        btn_save.clicked.connect(self._on_save_csv)
        self._btn_save = btn_save
        sl.addWidget(btn_save)

        # Device badge
        dev_lbl = QLabel(f"  Device: {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU'}")
        dev_lbl.setStyleSheet(
            f"color:{'#00E676' if GPU_AVAILABLE else PAL['text2']}; font-size:11px;"
            f"padding:4px 0; font-style:italic;")
        sl.addWidget(dev_lbl)
        sl.addStretch()
        
    #  MAIN TAB BUILD
    def _build_main_tabs(self):
        self._tabs.clear()
        self._canvases.clear()

        # Tab 0 Overview and Log
        ov_w = QWidget()
        ov_l = QVBoxLayout(ov_w)
        ov_l.setContentsMargins(12, 12, 12, 12); ov_l.setSpacing(10)

        hdr = QLabel(
            "<span style='font-size:20px;font-weight:800;color:#00E5FF;'>🔬 MoNuSeg Nuclei Segmentation</span><br>"
            "<span style='font-size:12px;color:#6B8FBD;'>Pipeline v2.1 · PyQt6 GUI · Biomedical Engineering ITS</span>")
        hdr.setTextFormat(Qt.TextFormat.RichText)
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(f"padding:16px; background:{PAL['bg3']}; border-radius:8px;"
                          f"border:1px solid {PAL['border']};")
        ov_l.addWidget(hdr)

        pipe_lbl = QLabel(
            "<b style='color:#42A5F5;'>Pipeline:</b><br>"
            "<span style='color:#B0C8E8;'>"
            "1. H&E Stain Extraction → 2. CLAHE Enhancement → 3. Gaussian Blur<br>"
            "→ 4. Adaptive Threshold → 5. Morphological Open → 6. Close + Hole Fill<br>"
            "→ 7. (Optional) Watershed → 8. Size Filter → IoU · Dice · Time"
            "</span>")
        pipe_lbl.setTextFormat(Qt.TextFormat.RichText)
        pipe_lbl.setStyleSheet(f"background:{PAL['bg3']}; padding:12px; border-radius:6px;"
                               f"border:1px solid {PAL['border2']}; line-height:1.6;")
        ov_l.addWidget(pipe_lbl)

        self._log_widget = LogWidget()
        self._log_widget.append_log("=== NucleiSeg GUI initialised ===")
        self._log_widget.append_log(f"  tifffile: {'✓' if _USE_TIFFFILE else '✗ (cv2 fallback)'}")
        self._log_widget.append_log(f"  GPU (CuPy): {'✓' if GPU_AVAILABLE else '✗'}")
        self._log_widget.append_log("\nSet dataset path in sidebar and click ▶ Run Processing.")
        ov_l.addWidget(self._log_widget, 1)

        self._prog_bar_ov = QProgressBar()
        self._prog_bar_ov.setRange(0, 4); self._prog_bar_ov.setValue(0)
        self._prog_bar_ov.setFixedHeight(6)
        ov_l.addWidget(self._prog_bar_ov)
        self._tabs.addTab(ov_w, "⚙  Overview")

        # Tabs 1-5 Per-image analysis
        tab_defs = [
            ("📊  Histograms",     "hist"),
            ("📡  CLAHE",          "clahe"),
            ("🧫  Segmentation",   "seg"),
            ("📈  Stage Analysis", "step"),
            ("⏱  Timing",         "timing"),
        ]
        for tab_label, key in tab_defs:
            outer_w  = QWidget()
            outer_l  = QVBoxLayout(outer_w)
            outer_l.setContentsMargins(6, 6, 6, 6)
            sub_tabs = QTabWidget()

            for name in DEFAULT_IMAGE_NAMES:
                short    = IMG_SHORT[name]
                tab_inner = QWidget()
                ti_l      = QVBoxLayout(tab_inner)
                ti_l.setContentsMargins(4, 4, 4, 4)
                ti_l.setSpacing(4)

                # Header row
                img_hdr = QWidget()
                img_hdr_l = QHBoxLayout(img_hdr)
                img_hdr_l.setContentsMargins(6, 2, 6, 2)
                name_lbl = QLabel(f"📌  {name}")
                name_lbl.setObjectName("imageTitle")
                img_hdr_l.addWidget(name_lbl)
                img_hdr_l.addStretch()

                # Metric badges for seg tab
                if key == "seg":
                    badge = MetricsBadge()
                    self._badges[name] = badge
                    img_hdr_l.addWidget(badge)

                ti_l.addWidget(img_hdr)

                # If step analysis, add two sub-sub-tabs, bar chart + grid
                if key == "step":
                    ss = QTabWidget()
                    ss.setStyleSheet("QTabBar::tab{padding:3px 8px;font-size:11px;}")
                    for sub_label, sub_key in [("📈 Bar Chart", f"step_bar_{name}"),
                                                ("🗂  Grid View",  f"step_grid_{name}")]:
                        cv = MatplotlibCanvas()
                        cv.show_placeholder(f"Run processing to see {sub_label.split()[1]} for\n{short}")
                        self._canvases[sub_key] = cv
                        ss.addTab(cv, sub_label)
                    ti_l.addWidget(ss, 1)
                else:
                    cv = MatplotlibCanvas()
                    cv.show_placeholder(f"Run processing to see {tab_label.strip()} for\n{short}")
                    ckey = f"{key}_{name}"
                    self._canvases[ckey] = cv
                    ti_l.addWidget(cv, 1)

                sub_tabs.addTab(tab_inner, short)

            # Cross-image comparison tabs
            if key in ("step", "timing"):
                cross_w = QWidget()
                cross_l = QVBoxLayout(cross_w)
                cross_l.setContentsMargins(4, 4, 4, 4)
                cv_cross = MatplotlibCanvas()
                cv_cross.show_placeholder("Run processing to see cross-image comparison")
                ckey = f"{key}_cross"
                self._canvases[ckey] = cv_cross
                cross_l.addWidget(cv_cross)
                sub_tabs.addTab(cross_w, "🔀  All Images")

            outer_l.addWidget(sub_tabs)
            self._tabs.addTab(outer_w, tab_label)

        # Tab 6 Results Summary
        sum_w = QWidget()
        sum_l = QVBoxLayout(sum_w)
        sum_l.setContentsMargins(6, 6, 6, 6)
        sum_tabs = QTabWidget()

        for slbl, skey in [("📊  Metrics",   "summary_metrics"),
                            ("⏱  Runtimes",  "summary_runtime"),
                            ("📋  Table",     "summary_table")]:
            sw = QWidget()
            sl_inner = QVBoxLayout(sw)
            sl_inner.setContentsMargins(4, 4, 4, 4)
            if skey != "summary_table":
                cv = MatplotlibCanvas()
                cv.show_placeholder("Run processing to view summary")
                self._canvases[skey] = cv
                sl_inner.addWidget(cv)
            else:
                self._results_table = QTextEdit()
                self._results_table.setReadOnly(True)
                self._results_table.setStyleSheet(
                    f"background:{PAL['bg0']}; color:{PAL['text0']};"
                    f"font-family:monospace; font-size:12px;"
                    f"border:1px solid {PAL['border2']};")
                self._results_table.setText("Run processing to view results table.")
                sl_inner.addWidget(self._results_table)
            sum_tabs.addTab(sw, slbl)

        sum_l.addWidget(sum_tabs)
        self._tabs.addTab(sum_w, "🏆  Summary")

    #  HELPERS
    def _set_status(self, msg: str):
        self._status.showMessage(msg)

    def _browse_base_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select MoNuSeg2018 Directory")
        if d:
            self._le_base.setText(d)

    def _browse_bg_image(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Background Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if f:
            self._le_bg.setText(f)
            self._try_load_bg()

    def _try_load_bg(self):
        """Load background image and push it to the central BgCentralWidget."""
        path = self._le_bg.text() if hasattr(self, '_le_bg') else "background.jpg"
        for candidate in [path, Path(__file__).parent / path]:
            p = Path(candidate)
            if p.exists():
                px = QPixmap(str(p))
                if not px.isNull():
                    self._central.set_bg_pixmap(px)
                    self._set_status(f"Background image loaded: {p.name}")
                    return
        # No valid image found → use gradient
        self._central.set_bg_pixmap(None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._central.update()   # repaint background on resize

    def _get_config(self) -> dict:
        """Collect all parameter values from sidebar widgets."""
        ksize = self._sp_gkern.value()
        if ksize % 2 == 0:
            ksize += 1   # ensure odd

        threshold_map  = {n: self._thr_cb[n].currentText()  for n in DEFAULT_IMAGE_NAMES}
        percentile_map = {n: self._thr_pct[n].value()       for n in DEFAULT_IMAGE_NAMES}

        params = {
            "use_clahe":        self._ck_clahe.isChecked(),
            "clahe_clip_limit": self._sp_clip.value(),
            "clahe_tile_size":  (self._sp_tile_w.value(), self._sp_tile_h.value()),
            "gaussian_ksize":   (ksize, ksize),
            "morph_kernel_size":self._sp_mks.value(),
            "open_iterations":  self._sp_oi.value(),
            "close_iterations": self._sp_ci.value(),
            "min_area_px":      self._sp_minA.value(),
            "max_area_px":      self._sp_maxA.value(),
            "peak_min_dist":    self._sp_pmd.value(),
            "dist_thresh_frac": self._sp_dtf.value(),
        }
        return {
            "base_dir":       self._le_base.text(),
            "image_names":    DEFAULT_IMAGE_NAMES,
            "stain_mode":     self._cb_stain.currentText(),
            "use_watershed":  self._ck_ws.isChecked(),
            "params":         params,
            "threshold_map":  threshold_map,
            "percentile_map": percentile_map,
            "run_diagnostics":self._ck_diag.isChecked(),
            "timing_repeats": self._sp_trep.value(),
        }

    #  RUN/WORKER SLOTS
    def _on_run(self):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._btn_run.setText("▶  Run Processing")
            self._btn_run.setEnabled(True)
            self._set_status("Processing aborted.")
            return

        cfg = self._get_config()
        self._results  = []
        self._step_all  = {}
        self._timing_all = {}
        self._log_widget.clear()
        self._log_widget.append_log(
            f"▶ Starting processing\n"
            f"  Base dir  : {cfg['base_dir']}\n"
            f"  Stain     : {cfg['stain_mode']}\n"
            f"  Watershed : {cfg['use_watershed']}\n"
            f"  CLAHE     : {cfg['params']['use_clahe']}  clip={cfg['params']['clahe_clip_limit']}\n"
            f"  Kernel    : {cfg['params']['gaussian_ksize']}\n"
            f"  Morph     : open={cfg['params']['open_iterations']}  close={cfg['params']['close_iterations']}"
        )
        self._prog_bar_ov.setValue(0)
        self._prog.setRange(0, len(DEFAULT_IMAGE_NAMES))
        self._prog.setValue(0)
        self._prog.setVisible(True)
        self._btn_run.setText("⏹  Stop")
        self._btn_save.setEnabled(False)
        self._set_status("Processing…")

        self._worker = ProcessingWorker(cfg)
        self._worker.log.connect(self._log_widget.append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.hist_ready.connect(self._on_hist_ready)
        self._worker.clahe_ready.connect(self._on_clahe_ready)
        self._worker.seg_ready.connect(self._on_seg_ready)
        self._worker.step_ready.connect(self._on_step_ready)
        self._worker.timing_ready.connect(self._on_timing_ready)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished_.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int, name: str):
        self._prog.setValue(current)
        self._prog_bar_ov.setValue(current)
        self._set_status(f"Processing [{current}/{total}]: {IMG_SHORT.get(name, name)}")

    def _on_hist_ready(self, name: str, image_rgb, H_u8):
        try:
            fig = create_histogram_figure(image_rgb, H_u8, IMG_SHORT.get(name, name))
            key = f"hist_{name}"
            if key in self._canvases:
                self._canvases[key].update_figure(fig)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Hist figure error: {e}")

    def _on_clahe_ready(self, name: str, H_u8_raw, params: dict):
        try:
            fig = create_clahe_figure(H_u8_raw, params, IMG_SHORT.get(name, name))
            key = f"clahe_{name}"
            if key in self._canvases:
                self._canvases[key].update_figure(fig)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ CLAHE figure error: {e}")

    def _on_seg_ready(self, name: str, image_rgb, gt_mask, pred_mask, metrics: dict):
        try:
            fig = create_segmentation_figure(image_rgb, gt_mask, pred_mask, metrics,
                                             IMG_SHORT.get(name, name))
            key = f"seg_{name}"
            if key in self._canvases:
                self._canvases[key].update_figure(fig)
            if name in self._badges:
                self._badges[name].update_metrics(metrics)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Seg figure error: {e}")

    def _on_step_ready(self, name: str, step_data: dict):
        try:
            short = IMG_SHORT.get(name, name)
            fig_bar = create_stepwise_bar_figure(
                step_data["stages"], step_data["ious"], step_data["dices"], short)
            bkey = f"step_bar_{name}"
            if bkey in self._canvases:
                self._canvases[bkey].update_figure(fig_bar)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Step bar figure error: {e}")

        try:
            # Grid figure uses the first image in seg, not stored here, but we can
            # create it from step_data alone (masks + stages)
            gkey = f"step_grid_{name}"
            if gkey in self._canvases:
                # We need image_rgb & gt_mask. Store in step_data for this purpose.
                if "image_rgb" in step_data and "gt_mask" in step_data:
                    fig_grid = create_stepwise_grid_figure(
                        step_data["image_rgb"], step_data["gt_mask"],
                        step_data["masks"], step_data["stages"],
                        step_data["ious"], step_data["dices"], IMG_SHORT.get(name, name))
                    self._canvases[gkey].update_figure(fig_grid)
                else:
                    self._canvases[gkey].show_placeholder(
                        "Grid needs image data — will be available after re-run\n"
                        "(upgrade: store image_rgb in step_data)")
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Step grid figure error: {e}")

        self._step_all[name] = step_data
        # Update cross-image view if all images done
        if len(self._step_all) == len(DEFAULT_IMAGE_NAMES):
            self._render_cross_step()

    def _on_timing_ready(self, name: str, timing_data: dict):
        try:
            fig = create_timing_figure(
                timing_data["stages"], timing_data["times_ms"],
                timing_data["total_ms"], timing_data["n_components"],
                IMG_SHORT.get(name, name))
            key = f"timing_{name}"
            if key in self._canvases:
                self._canvases[key].update_figure(fig)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Timing figure error: {e}")

        self._timing_all[name] = timing_data
        if len(self._timing_all) == len(DEFAULT_IMAGE_NAMES):
            self._render_cross_timing()

    def _render_cross_step(self):
        try:
            fig = create_cross_step_figure(self._step_all)
            if "step_cross" in self._canvases:
                self._canvases["step_cross"].update_figure(fig)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Cross-step figure error: {e}")

    def _render_cross_timing(self):
        try:
            fig = create_cross_timing_figure(self._timing_all)
            if "timing_cross" in self._canvases:
                self._canvases["timing_cross"].update_figure(fig)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Cross-timing figure error: {e}")

    def _on_all_done(self, results: list):
        self._results = results
        if not results:
            return

        df = pd.DataFrame(results)
        # Summary metrics figure
        try:
            fig_sum = create_summary_figure(df)
            if "summary_metrics" in self._canvases:
                self._canvases["summary_metrics"].update_figure(fig_sum)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Summary figure error: {e}")

        # Runtime figure
        try:
            fig_rt = create_runtime_figure(df)
            if "summary_runtime" in self._canvases:
                self._canvases["summary_runtime"].update_figure(fig_rt)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Runtime figure error: {e}")

        # Text table
        try:
            cols = ["Image","IoU","Dice","Precision","Recall","Running Time","Threshold","Pct"]
            cols = [c for c in cols if c in df.columns]
            table_str = (
                f"{'='*72}\n  RESULTS SUMMARY — MoNuSeg 2018  ·  v2.1\n{'='*72}\n\n"
                + df[cols].to_string(index=False)
                + f"\n\n{'─'*72}\n"
                + f"  Mean IoU   : {df['IoU'].mean():.4f}  (±{df['IoU'].std():.4f})\n"
                + f"  Mean Dice  : {df['Dice'].mean():.4f}  (±{df['Dice'].std():.4f})\n"
                + f"  Mean Time  : {df['Running Time'].mean():.4f} s\n"
                + f"{'='*72}\n"
            )
            self._results_table.setText(table_str)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Table error: {e}")

        self._btn_save.setEnabled(True)
        # Navigate to summary
        self._tabs.setCurrentIndex(6)

    def _on_error(self, msg: str):
        self._log_widget.append_log(f"\n⚠  ERROR: {msg}")
        self._set_status(f"Error: {msg}")

    def _on_finished(self):
        self._btn_run.setText("▶  Run Processing")
        self._btn_run.setEnabled(True)
        self._prog.setVisible(False)
        self._prog_bar_ov.setValue(4)
        self._set_status(f"✓  Processing complete — {len(self._results)} image(s) processed")

    #  SAVE CSV
    def _on_save_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results CSV", "results_v2.csv", "CSV Files (*.csv)")
        if path:
            pd.DataFrame(self._results).to_csv(path, index=False)
            self._set_status(f"Results saved → {path}")
            self._log_widget.append_log(f"\n  ✓ CSV saved: {path}")

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait(2000)
        plt.close("all")
        super().closeEvent(event)

#  ENTRY POINT
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    app.setStyle("Fusion")

    # Adjust palette for native elements
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(PAL["bg1"]))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(PAL["text0"]))
    pal.setColor(QPalette.ColorRole.Base,            QColor(PAL["bg4"]))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(PAL["bg3"]))
    pal.setColor(QPalette.ColorRole.Text,            QColor(PAL["text0"]))
    pal.setColor(QPalette.ColorRole.Button,          QColor(PAL["bg3"]))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(PAL["text0"]))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(PAL["accent1"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(PAL["text0"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(PAL["bg3"]))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(PAL["text0"]))
    app.setPalette(pal)

    win = NucleiSegApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
