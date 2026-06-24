"""Diagnostic data collectors: stepwise mask evolution and per-stage timing"""

import time
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes

from .stain    import get_h_channel
from .pipeline import adaptive_threshold, compute_metrics


def collect_stepwise_data(image_rgb, gt_mask, img_name, params,
                           threshold_map, percentile_map, stain_mode):
    """Run pipeline stage-by-stage; return raw data for plotting"""
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

    masks    = []
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

    s3 = adaptive_threshold(blur_s2, strategy, pct=pct_val)
    masks.append(s3)
    s4 = cv2.morphologyEx(s3, cv2.MORPH_OPEN, k, iterations=params["open_iterations"])
    masks.append(s4)
    s5 = cv2.morphologyEx(s4, cv2.MORPH_CLOSE, k, iterations=params["close_iterations"])
    s5 = (binary_fill_holes(s5 > 0) * 255).astype(np.uint8)
    masks.append(s5)

    s6 = s5.copy()
    nl, labs, sts, _ = cv2.connectedComponentsWithStats(s6, connectivity=8)
    areas = sts[:, cv2.CC_STAT_AREA]
    valid = np.where((areas >= params["min_area_px"]) & (areas <= params["max_area_px"]))[0]
    valid = valid[valid > 0]
    s6    = np.isin(labs, valid).astype(np.uint8) * 255
    masks.append(s6)

    ious  = [compute_metrics(m, gt_mask)["IoU"]  for m in masks]
    dices = [compute_metrics(m, gt_mask)["Dice"] for m in masks]
    return dict(
        stages=stages, masks=masks, ious=ious, dices=dices,
        image_rgb=image_rgb, gt_mask=gt_mask,
    )


def collect_timing_data(image_rgb, img_name, params, threshold_map,
                         percentile_map, stain_mode, n_repeats=3):
    """Time each pipeline stage; return raw data for plotting"""
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
