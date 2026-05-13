# core/signal_proc.py

import math
import numpy as np

class SignalPreprocessor:
    """Pipeline Stage F-1: IIR Biquad Bandpass Filter (Zero-Phase)."""
    
    def __init__(self, fps=30.0, low_hz=0.75, high_hz=2.5):
        self.fps = fps
        self.low_hz = low_hz
        self.high_hz = high_hz
        
        # Kalkulasi koefisien filter Biquad
        K = 2.0 * fps
        wl = 2.0 * fps * math.tan(math.pi * low_hz / fps)
        wh = 2.0 * fps * math.tan(math.pi * high_hz / fps)
        w0 = math.sqrt(wl * wh)
        BW = wh - wl
        D = K**2 + BW * K + w0**2
        
        self.b0 = (BW * K) / D
        self.b1 = 0.0
        self.b2 = -(BW * K) / D
        
        self.a1 = (2 * w0**2 - 2 * K**2) / D
        self.a2 = (K**2 - BW * K + w0**2) / D
        
        print(f"[SignalPreprocessor] IIR Biquad BPF [{low_hz}-{high_hz} Hz]")
        print(f"  b=[{self.b0:.6f}, 0, {self.b2:.6f}]  a=[1, {self.a1:.6f}, {self.a2:.6f}]")

    def _fwd(self, x):
        N = len(x)
        y = np.zeros(N)
        x1 = x2 = y1 = y2 = 0.0
        
        for n in range(N):
            yn = self.b0 * x[n] + self.b1 * x1 + self.b2 * x2 - self.a1 * y1 - self.a2 * y2
            x2 = x1
            x1 = x[n]
            y2 = y1
            y1 = yn
            y[n] = yn
            
        return y

    def filter(self, sig):
        if len(sig) < 6:
            return sig
        # Zero-phase filtering (forward-backward)
        y = self._fwd(sig - sig.mean())
        return self._fwd(y[::-1])[::-1]

    def normalize(self, sig):
        p = np.max(np.abs(sig))
        return sig / p if p >= 1e-10 else sig


class SGFilter:
    """Pipeline Stage F-2 & F-3: Savitzky-Golay smoothing (Manual Vandermonde)."""
    
    def __init__(self, window_len=9, poly_order=3):
        if window_len % 2 == 0:
            window_len += 1
        if poly_order >= window_len:
            raise ValueError("poly_order harus < window_len")
            
        self.window_len = window_len
        self.poly_order = poly_order
        self.half_w = window_len // 2
        self.kernel = self._compute_kernel(window_len, poly_order)
        
        print(f"[SGFilter] window={window_len}, poly={poly_order}, kernel_sum={self.kernel.sum():.6f}")

    def _compute_kernel(self, wl, po):
        half = wl // 2
        J = np.zeros((wl, po + 1), dtype=np.float64)
        
        for i in range(wl):
            xi = float(i - half)
            for k in range(po + 1):
                J[i, k] = xi**k
                
        JtJ = np.zeros((po + 1, po + 1), dtype=np.float64)
        for a in range(po + 1):
            for b in range(po + 1):
                s = 0.0
                for i in range(wl):
                    s += J[i, a] * J[i, b]
                JtJ[a, b] = s
                
        JtJ_inv = self._gauss_inv(JtJ)
        
        h = np.zeros(wl, dtype=np.float64)
        for i in range(wl):
            for k in range(po + 1):
                h[i] += JtJ_inv[0, k] * J[i, k]
                
        return h

    def _gauss_inv(self, A):
        n = A.shape[0]
        aug = np.hstack([A.copy(), np.eye(n, dtype=np.float64)])
        
        for col in range(n):
            mr = col + int(np.argmax(np.abs(aug[col:, col])))
            if mr != col:
                aug[[col, mr]] = aug[[mr, col]]
            p = aug[col, col]
            if abs(p) < 1e-14:
                raise ValueError("[SGFilter] JtJ singular")
            aug[col] /= p
            
            for row in range(n):
                if row != col:
                    aug[row] -= aug[row, col] * aug[col]
                    
        return aug[:, n:]

    def _convolve(self, signal):
        N = len(signal)
        hw = self.half_w
        padded = np.empty(N + 2 * hw, dtype=np.float64)
        
        # Edge padding dengan reflection
        padded[hw:hw + N] = signal
        padded[:hw] = signal[1:hw + 1][::-1]
        padded[hw + N:] = signal[N - hw - 1:N - 1][::-1]
        
        out = np.zeros(N, dtype=np.float64)
        wl = self.window_len
        for n in range(N):
            s = 0.0
            for k in range(wl):
                s += self.kernel[k] * padded[n + k]
            out[n] = s
            
        return out

    def smooth(self, signal):
        if len(signal) < self.window_len:
            return signal
        return self._convolve(signal)