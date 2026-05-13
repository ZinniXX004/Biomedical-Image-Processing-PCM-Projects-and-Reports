import time
import numpy as np
from skimage import exposure
from skimage.morphology import closing, disk, skeletonize
from skimage.metrics import (
    structural_similarity as _ssim_func,
    peak_signal_noise_ratio as _psnr_func,
    mean_squared_error as _mse_func,
)
from skimage.measure import shannon_entropy as _shannon_entropy

from config import KERNELS, KIRSCH_K

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
        
        # Normalize to [0.0, 1.0] to match standard image processing scales
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
    
    # Slide the kernel over the image (vectorized slice addition)
    for i in range(k_h):
        for j in range(k_w):
            out += padded[i:i+image.shape[0], j:j+image.shape[1]] * kernel[i, j]
            
    return out

def manual_gaussian_filter(image, sigma):
    """
    Manual Gaussian Blur utilizing Mathematical Separability.
    Instead of an N x N convolution, we apply a 1D horizontal pass 
    followed by a 1D vertical pass.
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

    # 2. Restoration / Denoising (Using manual math Gaussian)
    denoised = manual_gaussian_filter(enhanced, sigma=sigma) 
    morpho_pre = closing(denoised, disk(1))

    # 3. Gradient Computation (Using manual math 2D convolution)
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