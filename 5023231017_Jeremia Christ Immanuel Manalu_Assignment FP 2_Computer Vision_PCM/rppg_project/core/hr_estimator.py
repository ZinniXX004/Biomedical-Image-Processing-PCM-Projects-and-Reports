# core/hr_estimator.py

import math
import numpy as np
from collections import deque
from scipy.signal import find_peaks

class KalmanBPM:
    """Pipeline Stage I-2: 2-State Constant Velocity Kalman Filter for HR Tracking."""
    
    def __init__(self, q_bpm=0.5, q_trend=0.1, r_meas=8.0, dt=0.5):
        self.dt = dt
        self.F = np.array([[1.0, dt],[0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.Q = np.array([[q_bpm, 0.0], [0.0, q_trend]], dtype=np.float64)
        self.R = float(r_meas)
        self.x = None
        self.P = None
        self._init = False
        print(f"[KalmanBPM] q_bpm={q_bpm}, q_trend={q_trend}, r={r_meas}, dt={dt}s")

    def _init_state(self, bpm):
        self.x = np.array([[float(bpm)], [0.0]], dtype=np.float64)
        self.P = np.array([[self.R * 2, 0.0], [0.0, 1.0]], dtype=np.float64)
        self._init = True

    def update(self, z_bpm, confidence=1.0):
        if not self._init:
            self._init_state(z_bpm)
            return z_bpm
            
        # Predict step
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Adaptive Measurement Noise
        R_a = self.R / max(float(confidence), 0.01)
        z = np.array([[float(z_bpm)]])
        
        # Update step
        y_inn = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + np.array([[R_a]])
        K = P_pred @ self.H.T * (1.0 / S[0, 0])
        
        self.x = x_pred + K * y_inn[0, 0]
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        
        return float(self.x[0, 0])

    def reset(self):
        self.x = None
        self.P = None
        self._init = False
        print("[KalmanBPM] reset.")

    @property
    def bpm_estimate(self):
        return float(self.x[0, 0]) if self._init else None

    @property
    def trend(self):
        return float(self.x[1, 0]) if self._init else 0.0


class HeartRateEstimator:
    """Pipeline Stage H & I: Frequency Analysis (FFT) and HR Calculation + Kalman Tracking."""
    
    def __init__(self, fps=30.0, bpm_min=45.0, bpm_max=150.0,
                 conf_threshold=0.05, stability_window=8,
                 kalman_q_bpm=0.5, kalman_q_trend=0.1, kalman_r=8.0):
        self.fps = fps
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.conf_threshold = conf_threshold
        
        self._bpm_hist = deque(maxlen=stability_window)
        self._conf_hist = deque(maxlen=stability_window)
        
        dt_upd = 15 / fps
        self.kalman = KalmanBPM(
            q_bpm=kalman_q_bpm, 
            q_trend=kalman_q_trend, 
            r_meas=kalman_r, 
            dt=dt_upd
        )

    def _hanning(self, N):
        return np.array([0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)])

    def _dft_naive(self, x):
        N = len(x)
        X = np.zeros(N, dtype=complex)
        c = 2 * math.pi / N
        for k in range(N):
            for n in range(N):
                a = c * k * n
                X[k] += x[n] * complex(math.cos(a), -math.sin(a))
        return X

    def _fft(self, x):
        N = len(x)
        if N == 1:
            return np.array([complex(x[0], 0)])
        if N % 2 != 0:
            return self._dft_naive(x)
            
        E = self._fft(x[0::2])
        O = self._fft(x[1::2])
        h = N // 2
        tw = np.array([complex(math.cos(-2 * math.pi * k / N), math.sin(-2 * math.pi * k / N)) for k in range(h)])
        
        T = tw * O
        X = np.empty(N, dtype=complex)
        X[:h] = E + T
        X[h:] = E - T
        return X

    def _next_pow2(self, n):
        p = 1
        while p < n:
            p <<= 1
        return p

    def _fft_mags(self, sig):
        N = len(sig)
        Nf = self._next_pow2(N)
        p = np.zeros(Nf)
        p[:N] = sig * self._hanning(N)
        X = self._fft(p)
        half = Nf // 2
        freqs = np.array([k * self.fps / Nf for k in range(half)])
        return freqs, np.abs(X[:half])

    def estimate_fft(self, sig):
        if len(sig) < 8:
            return self._empty()
            
        freqs, mags = self._fft_mags(sig)
        v = (freqs >= self.bpm_min / 60) & (freqs <= self.bpm_max / 60)
        
        if not v.any():
            return self._empty()
            
        mv, fv = mags[v], freqs[v]
        pi = np.argmax(mv)
        pf = fv[pi]
        bpm = pf * 60
        
        conf = float(mv[pi]) / (float(mv.sum()) + 1e-8)
        bpm_med = self._iqr_median(bpm, conf)
        
        bpm_k = None
        if bpm_med is not None:
            bpm_k = self.kalman.update(bpm_med, confidence=conf)
            bpm_k = float(np.clip(bpm_k, self.bpm_min, self.bpm_max))
            
        return {
            'bpm': float(bpm),
            'bpm_median': float(bpm_med) if bpm_med else None,
            'bpm_kalman': bpm_k,
            'frequencies': freqs,
            'magnitudes': mags,
            'peak_freq': float(pf),
            'confidence': conf,
            'kalman_trend': self.kalman.trend
        }

    def estimate_peaks(self, sig):
        if len(sig) < 10:
            return {'bpm': None, 'peak_locs':[], 'ibi_mean': None, 'n_peaks': 0}
            
        md = max(int(self.fps / (self.bpm_max / 60)), 1)
        ar = np.max(sig) - np.min(sig)
        pks, _ = find_peaks(sig, distance=md, prominence=0.05 * ar)
        
        if len(pks) < 2:
            return {'bpm': None, 'peak_locs': pks.tolist(), 'ibi_mean': None, 'n_peaks': len(pks)}
            
        ibi = np.mean(np.diff(pks)) / self.fps
        bpm = 60 / ibi
        
        if not (self.bpm_min <= bpm <= self.bpm_max):
            bpm = None
            
        return {
            'bpm': float(bpm) if bpm else None,
            'peak_locs': pks.tolist(),
            'ibi_mean': float(ibi),
            'n_peaks': len(pks)
        }

    def _iqr_median(self, bpm, conf):
        if conf >= self.conf_threshold:
            self._bpm_hist.append(bpm)
            self._conf_hist.append(max(conf, 1e-6))
            
        if len(self._bpm_hist) == 0:
            return None
        if len(self._bpm_hist) == 1:
            return float(self._bpm_hist[0])
            
        ba, ca = np.array(self._bpm_hist), np.array(self._conf_hist)
        q1, q3 = np.percentile(ba, 25), np.percentile(ba, 75)
        iqr = q3 - q1
        mask = (ba >= q1 - 1.5 * iqr) & (ba <= q3 + 1.5 * iqr)
        
        bc, cc = (ba[mask], ca[mask]) if mask.any() else (ba, ca)
        ws = cc / cc.sum() if cc.sum() > 1e-8 else np.ones(len(cc)) / len(cc)
        
        si = np.argsort(bc)
        mid = min(int(np.searchsorted(np.cumsum(ws[si]), 0.5)), len(bc) - 1)
        return float(bc[si[mid]])

    def _empty(self):
        return {
            'bpm': None, 
            'bpm_median': None, 
            'bpm_kalman': None,
            'frequencies': np.array([]), 
            'magnitudes': np.array([]),
            'peak_freq': None,
            'confidence':0.0,'kalman_trend':0.0
            }
            
    def reset(self): 
        self._bpm_hist.clear(); self._conf_hist.clear(); self.kalman.reset()