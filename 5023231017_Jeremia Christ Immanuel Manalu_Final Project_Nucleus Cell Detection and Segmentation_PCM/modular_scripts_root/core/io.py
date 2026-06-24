"""Image I/O and MoNuSeg XML annotation parsing"""

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2

warnings.filterwarnings("ignore")

try:
    import tifffile
    _USE_TIFFFILE = True
except ImportError:
    _USE_TIFFFILE = False


def load_image(img_path: Path) -> np.ndarray:
    """Load TIFF/image → uint8 RGB (H, W, 3)"""
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
    """Parse MoNuSeg XML annotation → binary ground-truth mask"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
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
    return mask
