"""Core segmentation pipeline: thresholding, morphology, watershed, metrics"""

import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import threshold_multiotsu

from .stain import get_h_channel
from ..config import DEFAULT_PARAMS

GPU_AVAILABLE = False
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpnd
    cp.zeros(1)
    GPU_AVAILABLE = True
except Exception:
    pass


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
    else:
        _, binary = cv2.threshold(blur_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def segment_nuclei(image_rgb, img_name="", params=None,
                   use_gpu=False, use_ws=False,
                   threshold_map=None, percentile_map=None,
                   stain_mode="rgb2hed"):
    """Full V2.1 nucleus segmentation pipeline vectorized"""
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
        H_u8 = clahe.apply(H_u8)

    blur     = cv2.GaussianBlur(H_u8, params["gaussian_ksize"], 0)
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
            try:
                lbls = watershed(-dn, mkrs, mask=bb, compactness=0.001)
            except Exception:
                lbls = watershed(-dn, mkrs, mask=bb)
            binary = (lbls > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    valid = np.where((areas >= params["min_area_px"]) & (areas <= params["max_area_px"]))[0]
    valid = valid[valid > 0]
    binary = np.isin(labels, valid).astype(np.uint8) * 255
    return binary


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Pixel-level IoU, Dice, Precision, Recall, TP/FP/FN/TN"""
    p  = pred.astype(bool)
    g  = gt.astype(bool)
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
