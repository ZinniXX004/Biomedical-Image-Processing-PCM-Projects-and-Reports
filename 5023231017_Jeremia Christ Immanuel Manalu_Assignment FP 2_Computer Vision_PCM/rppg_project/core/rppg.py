# core/rppg.py

import numpy as np

class rPPGProcessor:
    """Pipeline Stage G: rPPG Extraction (CHROM/Green)."""
    
    def __init__(self, method='chrom'):
        if method not in ('chrom', 'green'): 
            raise ValueError(f"Method '{method}' tidak didukung.")
        self.method = method
        print(f"[rPPGProcessor] Metode: {method.upper()}")

    def compute(self, signals):
        R = signals['R'].astype(np.float64)
        G = signals['G'].astype(np.float64)
        B = signals['B'].astype(np.float64)
        
        if len(R) < 10: 
            return np.zeros(len(R))
            
        if self.method == 'green': 
            return G - G.mean()
            
        # CHROM Algorithm (De Haan and Jeanne, 2013)
        eps = 1e-8
        lum = R + G + B + eps
        Rn, Gn, Bn = R/lum, G/lum, B/lum
        
        Xs = 3*Rn - 2*Gn
        Ys = 1.5*Rn + Gn - 1.5*Bn
        
        # Alpha Projection
        return Xs - (np.std(Xs) / (np.std(Ys) + eps)) * Ys