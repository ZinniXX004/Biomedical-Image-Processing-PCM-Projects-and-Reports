"""Global constants and default pipeline configuration"""

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

DEFAULT_THRESHOLD_MAP  = {n: "percentile" for n in DEFAULT_IMAGE_NAMES}
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
    "morph_kernel_size": 3,
    "open_iterations":  2,
    "close_iterations": 0,
    "min_area_px":      20,
    "max_area_px":      80000,
    "peak_min_dist":    8,
    "dist_thresh_frac": 0.28,
}
