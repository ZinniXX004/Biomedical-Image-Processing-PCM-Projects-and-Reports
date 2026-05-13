import time
import numpy as np
from core.math_ops import manual_gaussian_filter, manual_convolve2d, normalize

def laplacian_sharpening(gray: np.ndarray, weight: float = 1.0, kernel_type: str = "H4") -> dict:
    t0 = time.perf_counter()
    blur = manual_gaussian_filter(gray, sigma=1.0)

    HL_x = np.array([[1.0, -2.0, 1.0]])
    HL_y = np.array([[1.0], [-2.0], [1.0]])
    lap_x = manual_convolve2d(blur, HL_x)
    lap_y = manual_convolve2d(blur, HL_y)
    lap_xy_sep = lap_x + lap_y
    sharp_sep = np.clip(gray - weight * lap_xy_sep, 0.0, 1.0)

    if kernel_type == "H8":
        HL = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    elif kernel_type == "H12":
        HL = np.array([[1., 2., 1.], [2., -12., 2.], [1., 2., 1.]])
    else:
        HL = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])

    lap_full = manual_convolve2d(blur, HL)
    sharp_full = np.clip(gray - weight * lap_full, 0.0, 1.0)

    return dict(
        blur=blur, lap_x=normalize(np.abs(lap_x)), lap_y=normalize(np.abs(lap_y)),
        lap_xy_sep=normalize(np.abs(lap_xy_sep)), sharp_sep=sharp_sep,
        lap_full=normalize(np.abs(lap_full)), sharp_full=sharp_full,
        elapsed=(time.perf_counter() - t0) * 1000,
    )

def unsharp_masking(gray: np.ndarray, a: float = 0.7, sigma: float = 1.0) -> dict:
    t0 = time.perf_counter()
    blurred = manual_gaussian_filter(gray, sigma=sigma)
    mask = gray - blurred
    sharpened = np.clip(gray + a * mask, 0.0, 1.0)
    return dict(
        blurred=blurred, mask=normalize(mask), sharpened=sharpened,
        elapsed=(time.perf_counter() - t0) * 1000,
    )