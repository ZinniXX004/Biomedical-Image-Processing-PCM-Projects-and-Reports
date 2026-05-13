import numpy as np
from skimage import exposure
from core.enhancement import contrast_stretching
from core.edge_detection import compute_traditional_pipeline, canny_scratch, canny_library
from core.sharpening import laplacian_sharpening, unsharp_masking

def compute_full_pipeline(gray, method, sigma, thr, enh_method, clahe_clip, cs_low, cs_high):
    """Wrapper that manages enhancement and passes to traditional pipeline."""
    acq = np.clip(gray.copy(), 0.0, 1.0)
    if enh_method == "CLAHE": enhanced = exposure.equalize_adapthist(acq, clip_limit=clahe_clip)
    elif enh_method == "HE": enhanced = exposure.equalize_hist(acq)
    elif enh_method == "CS": enhanced = contrast_stretching(acq, cs_low, cs_high)
    else: enhanced = acq.copy()

    res = compute_traditional_pipeline(enhanced, method, sigma, thr)
    res["acq"] = acq
    res["enhanced"] = enhanced
    return res

def compute_all_methods(gray, sigma, thr, canny_sigma, canny_tlo, canny_thi,
                        lap_weight, lap_kernel, usm_a, usm_sigma,
                        enh_method, clahe_clip, cs_low, cs_high) -> dict:
    """Run ALL edge detection + sharpening methods for benchmarking."""
    results = {}
    for m in ["Prewitt", "Sobel", "Roberts", "Extended Sobel", "Kirsch"]:
        r = compute_full_pipeline(gray, m, sigma, thr, enh_method, clahe_clip, cs_low, cs_high)
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