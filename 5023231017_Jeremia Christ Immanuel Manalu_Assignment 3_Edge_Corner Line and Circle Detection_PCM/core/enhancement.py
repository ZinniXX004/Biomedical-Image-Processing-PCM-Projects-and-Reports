import numpy as np
from skimage import exposure
from skimage.metrics import (
    structural_similarity as _ssim_func,
    peak_signal_noise_ratio as _psnr_func,
    mean_squared_error as _mse_func,
)
from skimage.measure import shannon_entropy as _shannon_entropy

def contrast_stretching(img, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    p2 = np.percentile(img, p_low)
    p98 = np.percentile(img, p_high)
    return np.clip((img - p2) / (p98 - p2 + 1e-9), 0.0, 1.0)

def compute_all_enhancements(acq: np.ndarray, clahe_clip: float = 0.03, cs_low: float = 2.0, cs_high: float = 98.0) -> dict:
    return {
        "CS":    contrast_stretching(acq, cs_low, cs_high),
        "HE":    exposure.equalize_hist(acq),
        "CLAHE": exposure.equalize_adapthist(acq, clip_limit=clahe_clip),
    }

def compute_enhancement_metrics(reference: np.ndarray, enhanced: np.ndarray) -> dict:
    mse = float(_mse_func(reference, enhanced))
    rmse = float(np.sqrt(mse))
    psnr = float(_psnr_func(reference, enhanced, data_range=1.0))
    if np.isinf(psnr) or np.isnan(psnr): psnr = 100.0
    ssim = float(_ssim_func(reference, enhanced, data_range=1.0))
    entropy = float(_shannon_entropy(enhanced))
    return {"RMSE": rmse, "PSNR": psnr, "SSIM": ssim, "Entropy": entropy}