"""Hematoxylin-channel extraction: rgb2hed, manual OD, and Macenko SVD"""

import numpy as np
import cv2
from skimage.color import rgb2hed as _skimage_rgb2hed


def extract_h_channel_rgb2hed(image_rgb: np.ndarray):
    hed  = _skimage_rgb2hed(image_rgb.astype(np.float32) / 255.0)
    H_raw = hed[:, :, 0]
    H_u8  = cv2.normalize(H_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return H_u8, H_raw


def extract_h_channel_manual(image_rgb: np.ndarray):
    img_float = np.clip(image_rgb.astype(np.float64) / 255.0, 1e-6, 1.0)
    OD = -np.log10(img_float)
    M  = np.array([[0.65, 0.70, 0.29],
                   [0.07, 0.99, 0.11],
                   [0.27, 0.57, 0.78]])
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
    ref_H  = np.array([0.6442, 0.7166, 0.2668])
    ref_H /= np.linalg.norm(ref_H)
    H_vec, E_vec = (
        (vec1, vec2) if np.dot(vec1, ref_H) >= np.dot(vec2, ref_H) else (vec2, vec1)
    )
    return H_vec, E_vec


def extract_h_channel_macenko(image_rgb: np.ndarray):
    H_vec, E_vec = estimate_stains_macenko(image_rgb)
    if H_vec is None:
        return extract_h_channel_manual(image_rgb)
    R_vec = np.cross(H_vec, E_vec)
    R_vec /= np.linalg.norm(R_vec) + 1e-12
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
    if mode == "macenko":
        return extract_h_channel_macenko(image_rgb)
    elif mode == "manual":
        return extract_h_channel_manual(image_rgb)
    return extract_h_channel_rgb2hed(image_rgb)
