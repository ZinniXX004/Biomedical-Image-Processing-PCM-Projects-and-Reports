# monitor/plotter.py

import cv2
import numpy as np
import threading
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Matplotlib Global Font Configuration
matplotlib.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 13,
})

class MultiROIPlotter:
    """
    Agg-only live signal plotter (background thread render).
    Render 2400x1500px @ DPI=100, downscale 1440x900 for display cv2.
    """

    MODE_LABELS = {
        'A_forehead': 'Mode A - Forehead', 
        'B_cheek': 'Mode B - Cheek',
        'C_combined': 'Mode C - Combined'
    }
    
    MODE_HDR_C = {
        'A_forehead': '#ff6655', 
        'B_cheek': '#55aaff', 
        'C_combined': '#88ee55'
    }
    
    CH_COLORS = {
        'R': ((1.0, 0.55, 0.55), (1.0, 0.1, 0.1)), 
        'G': ((0.4, 0.9, 0.5), (0.1, 0.85, 0.2)),
        'B': ((0.45, 0.65, 1.0), (0.15, 0.4, 1.0))
    }
    
    RPPG_C = {
        'A_forehead': (1.0, 0.4, 0.4), 
        'B_cheek': (0.3, 0.65, 1.0), 
        'C_combined': (0.4, 0.95, 0.4)
    }
    
    PEAK_C = (1.0, 0.95, 0.2)
    FS = {'title': 11, 'axis': 10, 'tick': 9, 'legend': 8}

    def __init__(self, window_name='rPPG Multi-ROI Comparison (A|B|C)'):
        self.window_name = window_name
        self._init_agg(window_name)

    def _init_agg(self, window_name):
        self._agg_w = 2400
        self._agg_h = 1500
        self._disp_w = 1440
        self._disp_h = 900

        self.fig = Figure(figsize=(24, 15), dpi=100, facecolor='#12121e')
        self.canvas = FigureCanvasAgg(self.fig)
        self.axes = {}
        modes = ['A_forehead', 'B_cheek', 'C_combined']
        
        for col, mode in enumerate(modes):
            for row in range(4):
                ax = self.fig.add_subplot(4, 3, row * 3 + col + 1)
                ax.set_facecolor('#0c0c18')
                ax.tick_params(colors='#aaaaaa', labelsize=self.FS['tick'])
                for sp in ax.spines.values():
                    sp.set_edgecolor('#3a3a4a')
                self.axes[(row, col)] = ax
                
        self.fig.tight_layout(pad=2.0, h_pad=1.5, w_pad=1.2)

        self._render_lock = threading.Lock()
        self._plot_img = None
        self._render_busy = False

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self._disp_w, self._disp_h)
        print(f"[MultiROIPlotter] Mode: Agg HQ | {self._agg_w}x{self._agg_h}px -> {self._disp_w}x{self._disp_h}px")

    def update(self, data, fps):
        """Non-blocking: schedule rendering in background thread."""
        if self._render_busy:
            return  # skip if frame is still rendering
        self._render_busy = True
        threading.Thread(target=self._render_worker_agg, args=(data, fps), daemon=True).start()

    def show_if_ready(self):
        """Display the latest rendered image if available."""
        with self._render_lock:
            if self._plot_img is not None:
                cv2.imshow(self.window_name, self._plot_img)

    def _render_worker_agg(self, data, fps):
        try:
            self._render_subplots(data, fps)
            self.canvas.draw()
            buf = self.canvas.buffer_rgba()
            img = np.asarray(buf, dtype=np.uint8).reshape(self._agg_h, self._agg_w, 4)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            bgr_disp = cv2.resize(bgr, (self._disp_w, self._disp_h), interpolation=cv2.INTER_AREA)
            
            with self._render_lock:
                self._plot_img = bgr_disp
        except Exception as e:
            print(f'[MultiROIPlotter] render error: {e}')
        finally:
            self._render_busy = False

    def _render_subplots(self, data, fps):
        modes =['A_forehead', 'B_cheek', 'C_combined']
        for col, mode in enumerate(modes):
            d = data.get(mode)
            if not d:
                continue
            rppg = d.get('rppg_norm')
            if rppg is None or len(rppg) < 4:
                continue
                
            N = len(rppg)
            t = np.arange(N, dtype=np.float64) / fps - N / fps

            self._plot_rppg(self.axes[(0, col)], t, rppg,
                            d.get('peak_locs',[]),
                            d.get('bpm_kalman'), d.get('bpm_raw'),
                            d.get('kalman_trend', 0.0), mode)
                            
            for row, ch in enumerate(['R', 'G', 'B'], 1):
                self._plot_ch(self.axes[(row, col)], t,
                              d.get(f'raw_{ch.lower()}', np.array([])),
                              d.get(f'filt_{ch.lower()}', np.array([])),
                              ch, is_bottom=(row == 3))

    def _plot_rppg(self, ax, t, sig, peaks, bpm_k, bpm_r, trend, mode):
        ax.clear()
        ax.set_facecolor('#0c0c18')
        c = self.RPPG_C[mode]
        N = len(sig)
        
        ax.plot(t, sig, color=c, linewidth=1.2, label='rPPG (CHROM+SG)')
        ax.axhline(0, color='#333344', linewidth=0.6, linestyle='--')
        
        if peaks:
            pk = np.array(peaks)
            pk = pk[pk < N]
            ax.scatter(t[pk], sig[pk], color=self.PEAK_C, s=30, zorder=5, label='peaks')
            
        hdr = self.MODE_LABELS[mode]
        hc = self.MODE_HDR_C[mode]
        
        if bpm_k:
            ts = '^' if trend > 0.3 else 'v' if trend < -0.3 else '->'
            bstr = f' | Kalman:{bpm_k:.1f} BPM {ts}'
            if bpm_r: 
                bstr += f' (raw:{bpm_r:.1f})'
        else:
            bstr = ' | buffering...'
            
        ax.set_title(f'{hdr}{bstr}', color=hc, fontsize=self.FS['title'], pad=4, fontweight='bold')
        ax.set_ylabel('rPPG (a.u.)', color='#bbbbbb', fontsize=self.FS['axis'], labelpad=4)
        ax.legend(fontsize=self.FS['legend'], loc='upper left', facecolor='#1a1a2e', labelcolor='white', framealpha=0.7)
        self._style(ax, t)

    def _plot_ch(self, ax, t, raw, filt, ch, is_bottom=False):
        ax.clear()
        ax.set_facecolor('#0c0c18')
        rc, fc = self.CH_COLORS[ch]
        N2 = min(len(raw), len(filt), len(t))
        
        if N2 < 4:
            ax.set_title(f'Channel {ch} - waiting for data', color='#888888', fontsize=self.FS['title'])
            return
            
        t2 = t[:N2]
        rn = raw[:N2]
        rn = (rn - rn.mean()) / (rn.std() + 1e-8)
        fn = filt[:N2]
        fn = (fn - fn.mean()) / (fn.std() + 1e-8)
        
        ax.plot(t2, rn, color=rc, linewidth=0.8, alpha=0.5, label=f'{ch} raw')
        ax.plot(t2, fn, color=fc, linewidth=1.1, label=f'{ch} BPF+SG')
        ax.axhline(0, color='#222233', linewidth=0.5, linestyle='--')
        
        ax.set_title(f'Channel {ch}: raw vs BPF+SG (0.75-2.5 Hz)', color=fc, fontsize=self.FS['title'], pad=4)
        ax.set_ylabel(f'Channel {ch} (norm.)', color='#bbbbbb', fontsize=self.FS['axis'], labelpad=4)
        
        if is_bottom:
            ax.set_xlabel('Time (s, relative to now)', color='#bbbbbb', fontsize=self.FS['axis'], labelpad=5)
            
        ax.legend(fontsize=self.FS['legend'], loc='upper left', facecolor='#1a1a2e', labelcolor='white', framealpha=0.7)
        self._style(ax, t2)

    def _style(self, ax, t):
        ax.tick_params(colors='#aaaaaa', labelsize=self.FS['tick'])
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a4a')
        if len(t) > 1:
            ax.set_xlim(t[0], t[-1])

    def destroy(self):
        cv2.destroyWindow(self.window_name)