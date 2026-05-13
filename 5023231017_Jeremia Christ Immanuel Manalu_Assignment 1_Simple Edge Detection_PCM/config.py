import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Steam-inspired Colour Theme
BG_DARK   = "#1b2838" 
BG_PANEL  = "#2a475e" 
BG_CARD   = "#171a21" 
TEXT_COL  = "#c7d5e0" 
SUBTEXT   = "#8f98a0" 
GRID_COL  = "#415467" 

STAGE_COLORS = {
    "preprocessing": "#66c0f4",   
    "gradient":      "#a4d007",   
    "results":       "#e3a968",   
    "enhancement":   "#b4a6fb",   
    "restoration":   "#4f94bc",   
}

METHOD_COLORS = {
    "Prewitt":        "#66c0f4",
    "Sobel":          "#a4d007",
    "Roberts":        "#F38BA8",
    "Extended Sobel": "#b4a6fb",
    "Kirsch":         "#e3a968",
}

ENH_COLORS = {
    "CS":    "#66c0f4",
    "HE":    "#a4d007",
    "CLAHE": "#b4a6fb",
}

EDGE_CMAP = LinearSegmentedColormap.from_list(
    "edge_glow",["#000000", "#171a21", "#1b2838", "#66c0f4", "#ffffff"]
)

# Kernel
KERNELS = {
    "Prewitt": {
        "Gx": np.array([[-1, 0, 1],[-1, 0, 1], [-1, 0, 1]], dtype=np.float64),
        "Gy": np.array([[-1,-1,-1], [ 0, 0, 0],[ 1, 1, 1]], dtype=np.float64),
    },
    "Sobel": {
        "Gx": np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]], dtype=np.float64),
        "Gy": np.array([[-1,-2,-1], [ 0, 0, 0],[ 1, 2, 1]], dtype=np.float64),
    },
    "Roberts": {
        "Gx": np.array([[ 1,  0], [ 0, -1]], dtype=np.float64),
        "Gy": np.array([[ 0,  1],[-1,  0]], dtype=np.float64),
    },
    "Extended Sobel": {
        "Gx": np.array([[-1, -2,  0,  2,  1],[-4, -8,  0,  8,  4],[-6,-12,  0, 12,  6],[-4, -8,  0,  8,  4],[-1, -2,  0,  2,  1]], dtype=np.float64),
        "Gy": np.array([[-1, -4, -6, -4, -1],[-2, -8,-12, -8, -2],[ 0,  0,  0,  0,  0],[ 2,  8, 12,  8,  2],[ 1,  4,  6,  4,  1]], dtype=np.float64),
    },
    "Kirsch": None,
}

KIRSCH_K = {
    "N":  np.array([[ 5,  5,  5], [-3,  0, -3], [-3, -3, -3]], dtype=np.float64),
    "NE": np.array([[-3,  5,  5], [-3,  0,  5], [-3, -3, -3]], dtype=np.float64),
    "E":  np.array([[-3, -3,  5],[-3,  0,  5], [-3, -3,  5]], dtype=np.float64),
    "SE": np.array([[-3, -3, -3],[-3,  0,  5], [-3,  5,  5]], dtype=np.float64),
    "S":  np.array([[-3, -3, -3],[-3,  0, -3], [ 5,  5,  5]], dtype=np.float64),
    "SW": np.array([[-3, -3, -3], [ 5,  0, -3],[ 5,  5, -3]], dtype=np.float64),
    "W":  np.array([[ 5, -3, -3], [ 5,  0, -3],[ 5, -3, -3]], dtype=np.float64),
    "NW": np.array([[ 5,  5, -3], [ 5,  0, -3],[-3, -3, -3]], dtype=np.float64),
}

KERNEL_INFO = {
    "Prewitt":        "Size: 3×3 | Kernels: 2 (Gx, Gy)\nWeights: uniform ±1 | Noise: medium",
    "Sobel":          "Size: 3×3 | Kernels: 2 (Gx, Gy)\nWeights: ±1,±2 center | Noise: med-high",
    "Roberts":        "Size: 2×2 | Kernels: 2 (diagonal)\nWeights: ±1 diagonal | Noise: low (fast)",
    "Extended Sobel": "Size: 5×5 | Kernels: 2 (Gx, Gy)\nWeights: ±1..±12 | Noise: high",
    "Kirsch":         "Size: 3×3 | Kernels: 8 (compass dirs)\nDirections: N,NE…NW | Noise: high",
}