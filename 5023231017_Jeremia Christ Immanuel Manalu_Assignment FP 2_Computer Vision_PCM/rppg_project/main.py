# main.py

"""
rPPG Heart Rate Monitor — PyQt6 GUI (Modernized UI/UX)
Entry point: python main.py

Dependencies (pip install -r requirements.txt):
  opencv-python, numpy, scipy, matplotlib, PyQt6

Project layout expected:
  main.py          <- this file
  config.py        <- DEFAULT_CONFIG dict
  chrom_justification.py <- Alat analisis CHROM
  core/            <- WebcamCapture, FaceEyeDetector, MultiROIExtractor,
                      SignalPreprocessor, SGFilter, rPPGProcessor,
                      KalmanBPM, HeartRateEstimator
  monitor/         <- MultiROIPlotter, MultiROIDataSaver, HRMonitor
  rppg_output/     <- auto-created for saved plots / CSV
"""

import sys
import time
import threading

import cv2
import numpy as np

from PyQt6.QtCore    import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui     import QImage, QPixmap, QFont, QColor, QPalette, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QSpinBox,
    QDoubleSpinBox, QComboBox, QLineEdit, QStatusBar, QSplitter,
    QScrollArea, QFrame, QSizePolicy, QProgressBar, QCheckBox
)

# Modules
try:
    from config import DEFAULT_CONFIG
    from core.webcam        import WebcamCapture
    from core.detector      import FaceEyeDetector
    from core.roi_extractor import MultiROIExtractor
    from core.signal_proc   import SignalPreprocessor, SGFilter
    from core.rppg          import rPPGProcessor
    from core.hr_estimator  import HeartRateEstimator
    from monitor.plotter    import MultiROIPlotter
    from monitor.data_saver import MultiROIDataSaver
    _MODULES_OK = True
except ImportError as _e:
    _MODULES_OK = False
    _IMPORT_ERR = str(_e)


PALETTE = {
    "bg":          "#0B0F19",   # Very dark navy/black background
    "panel":       "#131826",   # Sidebar / Panel background
    "card":        "#1A2133",   # Elevated cards
    "border":      "#2A3249",   # Soft borders
    "accent":      "#3B82F6",   # Vibrant Material Blue
    "accent2":     "#10B981",   # Emerald Green
    "text":        "#F3F4F6",   # Off-white / crisp text
    "text_dim":    "#9CA3AF",   # Cool Gray for secondary text
    "bpm_normal":  "#10B981",   # Emerald (60–100)
    "bpm_low":     "#3B82F6",   # Blue (<60)
    "bpm_high":    "#EF4444",   # Red (>100)
    "bpm_none":    "#4B5563",   # Dark Gray (Offline)
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
    font-family: 'Segoe UI', 'Roboto', 'Inter', 'Helvetica Neue', sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    margin-top: 14px;
    padding: 16px 12px 10px 12px;
    font-weight: 600;
    font-size: 12px;
    color: {PALETTE['text_dim']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 10px;
    background-color: {PALETTE['accent']};
    border-radius: 6px;
    color: white;
    font-weight: bold;
}}
QPushButton {{
    background-color: {PALETTE['card']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['border']};
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: #27314D;
    border-color: {PALETTE['accent']};
}}
QPushButton:pressed {{
    background-color: #1A2133;
}}
QPushButton:disabled {{
    color: #4B5563;
    background-color: {PALETTE['panel']};
    border-color: #1F2937;
}}
/* Spesifik Warna Tombol Modern */
QPushButton#btn_start {{
    background-color: #059669; color: white; border: none;
}}
QPushButton#btn_start:hover {{ background-color: #10B981; }}

QPushButton#btn_stop {{
    background-color: #DC2626; color: white; border: none;
}}
QPushButton#btn_stop:hover {{ background-color: #EF4444; }}

QPushButton#btn_reset {{
    background-color: #374151; color: white; border: none;
}}
QPushButton#btn_reset:hover {{ background-color: #4B5563; }}

QPushButton#btn_save {{
    background-color: #6366F1; color: white; border: none;
}}
QPushButton#btn_save:hover {{ background-color: #818CF8; }}

QPushButton#btn_justify {{
    background-color: #D97706; color: white; border: none;
}}
QPushButton#btn_justify:hover {{ background-color: #F59E0B; }}

QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {{
    background-color: {PALETTE['bg']};
    color: {PALETTE['text']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {PALETTE['accent']};
}}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {{
    border-color: {PALETTE['accent']};
}}
QScrollBar:vertical {{
    background: {PALETTE['bg']};
    width: 6px;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: #4B5563;
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {PALETTE['accent']};
}}
QProgressBar {{
    background-color: {PALETTE['bg']};
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    height: 12px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3B82F6, stop:1 #10B981);
    border-radius: 5px;
}}
QStatusBar {{
    background-color: {PALETTE['panel']};
    color: {PALETTE['text_dim']};
    font-size: 12px;
    border-top: 1px solid {PALETTE['border']};
}}
QLabel#section_title {{
    color: {PALETTE['text_dim']};
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px;
    border: 1px solid {PALETTE['border']};
    border-radius: 4px;
    background-color: {PALETTE['bg']};
}}
QCheckBox::indicator:checked {{
    background-color: {PALETTE['accent']};
    border-color: {PALETTE['accent']};
    image: url(check.png); /* Opsional jika ada icon */
}}
"""

class MonitorWorker(QThread):
    frame_ready    = pyqtSignal(np.ndarray)          
    bpm_updated    = pyqtSignal(dict)                
    status_changed = pyqtSignal(str)                 
    buffer_changed = pyqtSignal(int, int)            
    save_done      = pyqtSignal(str)                 
    error_occurred = pyqtSignal(str)                 

    MODES    =['A_forehead', 'B_cheek', 'C_combined']
    MODE_OSD = {
        'A_forehead': (80, 100, 255),
        'B_cheek':    (255, 160, 50),
        'C_combined': (60, 220, 80),
    }

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg          = cfg
        self._running     = False
        self._save_flag   = False
        self._reset_flag  = False
        self._results     = {m: None for m in self.MODES}
        self._bpm_log     = {m:[]   for m in self.MODES}
        self._hr_lock     = threading.Lock()
        self._hr_busy     = {m: False for m in self.MODES}
        self._frame_count = 0
        self._fps_display = 0.0
        self._fps_t_prev  = None

    def stop(self):
        self._running = False

    def request_save(self):
        self._save_flag = True

    def request_reset(self):
        self._reset_flag = True

    def get_raw_signals_snapshot(self, mode='C_combined'):
        with self._hr_lock:
            r = self._results.get(mode)
            if r is not None and 'raw_signals' in r:
                return {
                    'R': r['raw_signals']['R'].copy(),
                    'G': r['raw_signals']['G'].copy(),
                    'B': r['raw_signals']['B'].copy(),
                }
        return None

    def run(self):
        cfg = self.cfg
        fps = cfg["fps"]

        try:
            webcam    = WebcamCapture(camera_index=cfg["camera_index"], fps=int(fps))
            detector  = FaceEyeDetector()
            extractor = MultiROIExtractor(buffer_seconds=cfg["buffer_seconds"], fps=fps)
            preproc   = SignalPreprocessor(fps=fps, low_hz=cfg["bpf_low_hz"], high_hz=cfg["bpf_high_hz"])
            sg_filt   = SGFilter(window_len=cfg["sg_window"], poly_order=cfg["sg_poly"])
            rppg_proc = rPPGProcessor(method=cfg["rppg_method"])
            hr_ests   = {m: HeartRateEstimator(
                             fps=fps, bpm_min=cfg["bpm_min"], bpm_max=cfg["bpm_max"],
                             kalman_q_bpm=cfg["kalman_q_bpm"], kalman_q_trend=cfg["kalman_q_trend"],
                             kalman_r=cfg["kalman_r"]) for m in self.MODES}
            plotter   = MultiROIPlotter()
            saver     = MultiROIDataSaver(output_dir=cfg["output_dir"])
        except Exception as e:
            self.error_occurred.emit(f"Init error: {e}")
            return

        if not webcam.open():
            self.error_occurred.emit(f"Cannot open camera index {cfg['camera_index']}. Try a different index.")
            return

        self.status_changed.emit("Camera opened. Waiting for face…")
        self._running = True
        update_interval = cfg["update_interval"]
        plot_interval   = cfg["plot_interval"]
        buffer_max      = extractor.buffer_len

        try:
            while self._running:
                if self._reset_flag:
                    self._reset_flag = False
                    extractor.reset()
                    for m in self.MODES: hr_ests[m].reset()
                    with self._hr_lock:
                        self._results  = {m: None for m in self.MODES}
                        self._bpm_log  = {m:[] for m in self.MODES}
                        self._hr_busy  = {m: False for m in self.MODES}
                    self._frame_count = 0
                    self.bpm_updated.emit({m: {} for m in self.MODES})
                    self.status_changed.emit("Reset complete. Waiting for face…")

                if self._save_flag:
                    self._save_flag = False
                    data = {m: r for m, r in self._results.items() if r is not None}
                    if data:
                        self.status_changed.emit("Saving plots and CSV…")
                        out_dir = saver.save_all(data, fps, bpm_log=self._bpm_log)
                        self.save_done.emit(out_dir)
                        self.status_changed.emit(f"Saved to '{out_dir}/'")
                    else:
                        self.status_changed.emit("Nothing to save yet — buffer still filling.")

                ret, frame = webcam.read_frame()
                if not ret or frame is None:
                    continue
                self._frame_count += 1
                ts = time.time()

                if self._fps_t_prev and (dt := ts - self._fps_t_prev) > 0:
                    self._fps_display = (0.1 * (1 / dt) + 0.9 * self._fps_display)
                self._fps_t_prev = ts

                det  = detector.detect(frame)
                vis  = detector.draw(frame.copy(), det)
                face_ok = bool(det['faces'])

                if face_ok:
                    bbox = max(det['faces'], key=lambda f: f[2] * f[3])
                    vis  = extractor.draw(vis, bbox)
                    extractor.update(frame, bbox, ts)

                    if self._frame_count % update_interval == 0:
                        for mode in self.MODES:
                            if extractor.is_ready(mode, cfg["min_ready_sec"]) and not self._hr_busy[mode]:
                                sigs = extractor.get_signals(mode)
                                self._hr_busy[mode] = True
                                threading.Thread(
                                    target=self._compute_hr_async,
                                    args=(mode, sigs, preproc, sg_filt, rppg_proc, hr_ests[mode]),
                                    daemon=True).start()

                vis = self._draw_overlay(vis, face_ok, extractor.buffer_len, len(extractor.buffers['A_forehead']['R']))

                self.frame_ready.emit(vis)
                n = len(extractor.buffers['A_forehead']['R'])
                self.buffer_changed.emit(n, buffer_max)

                if self._frame_count % plot_interval == 0:
                    plot_data = {}
                    for mode in self.MODES:
                        with self._hr_lock:
                            r = self._results.get(mode)
                        if r is None: continue
                        ff = r['fft_result']
                        plot_data[mode] = {
                            'rppg_norm':    r['rppg_norm'],
                            'peak_locs':    r['peak_result'].get('peak_locs',[]),
                            'bpm_kalman':   ff.get('bpm_kalman'),
                            'bpm_raw':      ff.get('bpm'),
                            'kalman_trend': ff.get('kalman_trend', 0.0),
                            'raw_r':  r['raw_signals']['R'], 'filt_r': r['filt_r'],
                            'raw_g':  r['raw_signals']['G'], 'filt_g': r['filt_g'],
                            'raw_b':  r['raw_signals']['B'], 'filt_b': r['filt_b'],
                        }
                    if plot_data:
                        plotter.update(plot_data, fps)
                
                plotter.show_if_ready()
                cv2.waitKey(1) 

                if self._frame_count % update_interval == 0:
                    bpm_snapshot = {}
                    for mode in self.MODES:
                        with self._hr_lock:
                            r = self._results.get(mode)
                        if r is None:
                            bpm_snapshot[mode] = {}
                        else:
                            ff = r['fft_result']
                            bpm_snapshot[mode] = {
                                'bpm_kalman':   ff.get('bpm_kalman'),
                                'bpm_median':   ff.get('bpm_median'),
                                'bpm_raw':      ff.get('bpm'),
                                'confidence':   ff.get('confidence', 0.0),
                                'kalman_trend': ff.get('kalman_trend', 0.0),
                                'peak_bpm':     r['peak_result'].get('bpm'),
                            }
                    self.bpm_updated.emit(bpm_snapshot)

        except Exception as e:
            self.error_occurred.emit(f"Runtime error: {e}")
        finally:
            if cfg.get("auto_save"):
                data = {m: r for m, r in self._results.items() if r is not None}
                if data: saver.save_all(data, fps, bpm_log=self._bpm_log)
            plotter.destroy()
            webcam.release()
            cv2.destroyAllWindows()
            self.status_changed.emit("Monitor stopped.")

    def _compute_hr_async(self, mode, signals, preproc, sg_filt, rppg_proc, hr_est):
        try:
            result = self._compute_hr_from_signals(signals, preproc, sg_filt, rppg_proc, hr_est)
            log_entry = result.pop('_log_entry', None)
            with self._hr_lock:
                self._results[mode] = result
                if log_entry:
                    self._bpm_log[mode].append(log_entry)
        except Exception as e:
            print(f"[MonitorWorker] HR compute error ({mode}): {e}")
        finally:
            self._hr_busy[mode] = False

    def _compute_hr_from_signals(self, signals, preproc, sg_filt, rppg_proc, hr_est):
        r_bpf = preproc.filter(signals['R'])
        g_bpf = preproc.filter(signals['G'])
        b_bpf = preproc.filter(signals['B'])

        r_sg  = sg_filt.smooth(r_bpf)
        g_sg  = sg_filt.smooth(g_bpf)
        b_sg  = sg_filt.smooth(b_bpf)

        sigs_clean = {**signals, 'R': r_sg, 'G': g_sg, 'B': b_sg}
        rppg_raw   = rppg_proc.compute(sigs_clean)

        rppg_bpf  = preproc.filter(rppg_raw)
        rppg_sg   = sg_filt.smooth(rppg_bpf)
        rppg_norm = preproc.normalize(rppg_sg)

        fft_res  = hr_est.estimate_fft(rppg_norm)
        peak_res = hr_est.estimate_peaks(rppg_norm)

        log_entry = {
            'timestamp':   time.time(),
            'bpm_raw':     fft_res.get('bpm'),
            'bpm_median':  fft_res.get('bpm_median'),
            'bpm_kalman':  fft_res.get('bpm_kalman'),
            'confidence':  fft_res.get('confidence', 0),
        }

        return {
            'raw_signals': signals, 'rppg_norm': rppg_norm,
            'bpf_r': r_bpf, 'bpf_g': g_bpf, 'bpf_b': b_bpf,
            'filt_r': r_sg, 'filt_g': g_sg,  'filt_b': b_sg,
            'fft_result': fft_res, 'peak_result': peak_res,
            '_log_entry': log_entry,
        }

    def _draw_overlay(self, frame, face_ok, buffer_len, n):
        h, w = frame.shape[:2]
        ov   = frame.copy()
        cv2.rectangle(ov, (5, 5), (440, 235), (25, 25, 25), -1)
        cv2.addWeighted(ov, 0.4, frame, 0.6, 0, frame)

        cv2.putText(frame, 'rPPG HR Monitor v10 -- BPF+SG+Kalman',
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
        pct = min(n / max(buffer_len, 1), 1.0)
        cv2.putText(frame, f'FPS:{self._fps_display:.1f} | Buffer:{n}/{buffer_len}({pct*100:.0f}%)',
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, .40, (200, 200, 200), 1)
        cv2.rectangle(frame, (10, 50), (210, 58), (70, 70, 70), -1)
        cv2.rectangle(frame, (10, 50), (10 + int(200 * pct), 58), (0, 180, 100), -1)

        y0 = 85
        mshort = {'A_forehead': '[A]FH', 'B_cheek': '[B]CK', 'C_combined': '[C]CB'}
        for i, mode in enumerate(self.MODES):
            with self._hr_lock:
                r = self._results.get(mode)
            oc = self.MODE_OSD[mode]
            ms = mshort[mode]
            if r is None:
                cv2.putText(frame, f'{ms}: buffering…', (10, y0 + i * 48), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 100, 255), 1)
            else:
                ff    = r['fft_result']
                bk, bm, br = ff.get('bpm_kalman'), ff.get('bpm_median'), ff.get('bpm')
                conf, trend = ff.get('confidence', 0), ff.get('kalman_trend', 0)
                ts_sym = ('^' if trend > 0.3 else 'v' if trend < -0.3 else '->')
                
                if bk:
                    bpm_c = ((0, 255, 100) if 60 <= bk <= 100 else (0, 200, 255) if bk < 120 else (0, 50, 255))
                    cv2.putText(frame, f'{ms}: {bk:.1f} BPM {ts_sym}', (10, y0 + i * 48), cv2.FONT_HERSHEY_SIMPLEX, .72, bpm_c, 2)
                    detail = f'  conf={conf*100:.1f}%'
                    if bm: detail += f' | med={bm:.1f}'
                    if br: detail += f' | raw={br:.1f}'
                    cv2.putText(frame, detail, (10, y0 + i * 48 + 18), cv2.FONT_HERSHEY_SIMPLEX, .35, oc, 1)
                else:
                    cv2.putText(frame, f'{ms}: computing…', (10, y0 + i * 48), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 100, 255), 1)

        if not face_ok:
            cv2.putText(frame, 'Position face in frame', (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 100, 255), 1)
        cv2.putText(frame, 'Q=Quit  R=Reset  S=Save', (w - 205, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (150, 150, 150), 1)
        return frame

#  BPM CARD WIDGET
class BpmCard(QFrame):
    MODE_META = {
        'A_forehead': ('Forehead ROI',  '🧠'),
        'B_cheek':    ('Cheek ROI',     '😊'),
        'C_combined': ('Combined ROI',  '✨'),
    }

    def __init__(self, mode: str, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setObjectName("bpm_card")
        
        # UI Card Modern: transisi hover border, padding ekstra
        self.setStyleSheet(f"""
            QFrame#bpm_card {{
                background-color: {PALETTE['card']};
                border: 1px solid {PALETTE['border']};
                border-radius: 12px; 
            }}
            QFrame#bpm_card:hover {{
                border: 1px solid {PALETTE['accent']};
                background-color: #1F273D;
            }}
        """)

        label_text, icon = self.MODE_META.get(mode, (mode, ''))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(6)

        header = QLabel(f"{icon}  {label_text}")
        header.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size:12px; font-weight:700; letter-spacing:0.5px; text-transform: uppercase;")
        layout.addWidget(header)

        self.lbl_bpm = QLabel("— BPM")
        self.lbl_bpm.setStyleSheet(f"color: {PALETTE['bpm_none']}; font-size: 46px; font-weight: 800; letter-spacing: -1.5px;")
        self.lbl_bpm.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.lbl_bpm)

        # Container detail data biar rapi
        detail_container = QVBoxLayout()
        detail_container.setSpacing(2)
        
        self.lbl_detail = QLabel("Waiting for data…")
        self.lbl_detail.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 11px; font-weight:500;")
        self.lbl_detail.setWordWrap(True)
        detail_container.addWidget(self.lbl_detail)

        self.lbl_trend = QLabel("")
        self.lbl_trend.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {PALETTE['accent']};")
        detail_container.addWidget(self.lbl_trend)

        layout.addLayout(detail_container)
        layout.addStretch()

    def update_bpm(self, data: dict):
        if not data:
            self.lbl_bpm.setText("— BPM")
            self.lbl_bpm.setStyleSheet(f"color: {PALETTE['bpm_none']}; font-size: 46px; font-weight: 800; letter-spacing: -1.5px;")
            self.lbl_detail.setText("Buffering…")
            self.lbl_trend.setText("")
            return

        bk, bm, br, bp = data.get('bpm_kalman'), data.get('bpm_median'), data.get('bpm_raw'), data.get('peak_bpm')
        conf, trend = data.get('confidence', 0.0), data.get('kalman_trend', 0.0)

        if bk is not None:
            color = PALETTE['bpm_low'] if bk < 60 else PALETTE['bpm_normal'] if bk <= 100 else PALETTE['bpm_high']
            self.lbl_bpm.setText(f"{bk:.1f}")
            self.lbl_bpm.setStyleSheet(f"color: {color}; font-size: 46px; font-weight: 800; letter-spacing: -1.5px;")

            details =[]
            if bm is not None: details.append(f"Med: {bm:.1f}")
            if br is not None: details.append(f"Raw: {br:.1f}")
            if bp is not None: details.append(f"Pk: {bp:.1f}")
            details.append(f"Conf: {conf*100:.0f}%")
            self.lbl_detail.setText(" • ".join(details))

            arrow = ("▲ Up" if trend > 0.3 else "▼ Down" if trend < -0.3 else "→ Stable")
            t_color = "#10B981" if abs(trend) < 0.3 else "#EF4444"
            self.lbl_trend.setStyleSheet(f"font-size: 11px; font-weight: bold; color: {t_color};")
            self.lbl_trend.setText(f"Trend: {arrow} ({trend:+.2f}/s)")
        else:
            self.lbl_bpm.setText("—")
            self.lbl_bpm.setStyleSheet(f"color: {PALETTE['bpm_none']}; font-size: 46px; font-weight: 800; letter-spacing: -1.5px;")
            self.lbl_detail.setText("Computing…")
            self.lbl_trend.setText("")

#  SETTINGS PANEL
class SettingsPanel(QWidget):
    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260) # Sedikit dilebarkan untuk UI yang lebih bernapas
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        grp_cam = QGroupBox("📷 Camera Source")
        g = QGridLayout(grp_cam)
        g.setColumnStretch(1, 1)
        g.addWidget(QLabel("Index"), 0, 0)
        self.spin_cam = QSpinBox(); self.spin_cam.setRange(0, 10); self.spin_cam.setValue(cfg["camera_index"])
        g.addWidget(self.spin_cam, 0, 1)
        g.addWidget(QLabel("FPS"), 1, 0)
        self.spin_fps = QSpinBox(); self.spin_fps.setRange(10, 60); self.spin_fps.setValue(int(cfg["fps"]))
        g.addWidget(self.spin_fps, 1, 1)
        root.addWidget(grp_cam)

        grp_buf = QGroupBox("⏳ Buffer Settings")
        g2 = QGridLayout(grp_buf)
        g2.setColumnStretch(1, 1)
        g2.addWidget(QLabel("Buffer (s)"), 0, 0)
        self.spin_buf = QDoubleSpinBox(); self.spin_buf.setRange(5, 60); self.spin_buf.setSingleStep(1); self.spin_buf.setValue(cfg["buffer_seconds"])
        g2.addWidget(self.spin_buf, 0, 1)
        g2.addWidget(QLabel("Ready (s)"), 1, 0)
        self.spin_minr = QDoubleSpinBox(); self.spin_minr.setRange(2, 20); self.spin_minr.setSingleStep(0.5); self.spin_minr.setValue(cfg["min_ready_sec"])
        g2.addWidget(self.spin_minr, 1, 1)
        root.addWidget(grp_buf)

        grp_dsp = QGroupBox("🎛️ Signal Processing")
        g3 = QGridLayout(grp_dsp)
        g3.setColumnStretch(1, 1)
        g3.addWidget(QLabel("BPF Low"), 0, 0)
        self.spin_bpf_lo = QDoubleSpinBox(); self.spin_bpf_lo.setRange(0.3, 1.5); self.spin_bpf_lo.setSingleStep(0.05); self.spin_bpf_lo.setValue(cfg["bpf_low_hz"])
        g3.addWidget(self.spin_bpf_lo, 0, 1)
        g3.addWidget(QLabel("BPF High"), 1, 0)
        self.spin_bpf_hi = QDoubleSpinBox(); self.spin_bpf_hi.setRange(1.5, 4.0); self.spin_bpf_hi.setSingleStep(0.05); self.spin_bpf_hi.setValue(cfg["bpf_high_hz"])
        g3.addWidget(self.spin_bpf_hi, 1, 1)
        g3.addWidget(QLabel("SG Window"), 2, 0)
        self.spin_sgw = QSpinBox(); self.spin_sgw.setRange(5, 31); self.spin_sgw.setSingleStep(2); self.spin_sgw.setValue(cfg["sg_window"])
        g3.addWidget(self.spin_sgw, 2, 1)
        g3.addWidget(QLabel("SG Poly"), 3, 0)
        self.spin_sgp = QSpinBox(); self.spin_sgp.setRange(2, 6); self.spin_sgp.setValue(cfg["sg_poly"])
        g3.addWidget(self.spin_sgp, 3, 1)
        g3.addWidget(QLabel("Method"), 4, 0)
        self.combo_rppg = QComboBox(); self.combo_rppg.addItems(["chrom", "green"]); self.combo_rppg.setCurrentText(cfg["rppg_method"])
        g3.addWidget(self.combo_rppg, 4, 1)
        root.addWidget(grp_dsp)

        grp_kal = QGroupBox("🎯 Kalman Filter")
        g4 = QGridLayout(grp_kal)
        g4.setColumnStretch(1, 1)
        g4.addWidget(QLabel("q_bpm"), 0, 0)
        self.spin_kqb = QDoubleSpinBox(); self.spin_kqb.setRange(0.01, 5.0); self.spin_kqb.setSingleStep(0.1); self.spin_kqb.setDecimals(2); self.spin_kqb.setValue(cfg["kalman_q_bpm"])
        g4.addWidget(self.spin_kqb, 0, 1)
        g4.addWidget(QLabel("q_trend"), 1, 0)
        self.spin_kqt = QDoubleSpinBox(); self.spin_kqt.setRange(0.01, 2.0); self.spin_kqt.setSingleStep(0.05); self.spin_kqt.setDecimals(2); self.spin_kqt.setValue(cfg["kalman_q_trend"])
        g4.addWidget(self.spin_kqt, 1, 1)
        g4.addWidget(QLabel("r_meas"), 2, 0)
        self.spin_kr = QDoubleSpinBox(); self.spin_kr.setRange(1.0, 50.0); self.spin_kr.setSingleStep(1.0); self.spin_kr.setDecimals(1); self.spin_kr.setValue(cfg["kalman_r"])
        g4.addWidget(self.spin_kr, 2, 1)
        root.addWidget(grp_kal)

        grp_out = QGroupBox("💾 Output & Logs")
        g5 = QGridLayout(grp_out)
        g5.setColumnStretch(1, 1)
        g5.addWidget(QLabel("Folder"), 0, 0)
        self.edit_outdir = QLineEdit(cfg["output_dir"])
        g5.addWidget(self.edit_outdir, 0, 1)
        self.chk_autosave = QCheckBox("Auto-save on quit")
        self.chk_autosave.setChecked(cfg.get("auto_save", True))
        g5.addWidget(self.chk_autosave, 1, 0, 1, 2)
        root.addWidget(grp_out)

        root.addStretch()

    def get_config(self) -> dict:
        return {
            "camera_index":    self.spin_cam.value(),
            "fps":             float(self.spin_fps.value()),
            "buffer_seconds":  self.spin_buf.value(),
            "min_ready_sec":   self.spin_minr.value(),
            "update_interval": 15,
            "plot_interval":   15,
            "bpf_low_hz":      self.spin_bpf_lo.value(),
            "bpf_high_hz":     self.spin_bpf_hi.value(),
            "sg_window":       self.spin_sgw.value(),
            "sg_poly":         self.spin_sgp.value(),
            "rppg_method":     self.combo_rppg.currentText(),
            "bpm_min":         45.0,
            "bpm_max":         150.0,
            "kalman_q_bpm":    self.spin_kqb.value(),
            "kalman_q_trend":  self.spin_kqt.value(),
            "kalman_r":        self.spin_kr.value(),
            "output_dir":      self.edit_outdir.text().strip() or "rppg_output",
            "auto_save":       self.chk_autosave.isChecked(),
        }


class JustifyWorker(QThread):
    finished_plot = pyqtSignal(str, dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, cfg, real_data):
        super().__init__()
        self.cfg = cfg
        self.real_data = real_data

    def run(self):
        try:
            import matplotlib
            matplotlib.use('Agg') 

            print("\n[JustifyWorker] Memulai justifikasi CHROM dari background thread...")
            from chrom_justification import run_chrom_justification
            path, results = run_chrom_justification(self.cfg, self.real_data)
            
            print(f"[JustifyWorker] Analisis selesai! Menyimpan plot ke: {path}\n")
            self.finished_plot.emit(path, results)
        except Exception as e:
            import traceback
            traceback.print_exc() 
            self.error_occurred.emit(str(e))

#  MAIN WINDOW 
class MainWindow(QMainWindow):
    MODES =['A_forehead', 'B_cheek', 'C_combined']

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical-Grade rPPG Heart Rate Monitor")
        self.resize(1300, 850)
        self.setMinimumSize(960, 650)
        self.setStyleSheet(STYLESHEET)

        self._worker: MonitorWorker | None = None

        cfg = DEFAULT_CONFIG.copy() if _MODULES_OK else {
            "camera_index": 0, "fps": 30.0, "buffer_seconds": 10.0,
            "min_ready_sec": 5.0, "update_interval": 15, "plot_interval": 15,
            "bpf_low_hz": 0.75, "bpf_high_hz": 2.5, "sg_window": 9,
            "sg_poly": 3, "rppg_method": "chrom", "bpm_min": 45.0,
            "bpm_max": 150.0, "kalman_q_bpm": 0.5, "kalman_q_trend": 0.1,
            "kalman_r": 8.0, "output_dir": "rppg_output", "auto_save": True,
        }

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # ── KIRI: Settings ──
        left_col = QVBoxLayout()
        title_app = QLabel("rPPG SYSTEM")
        title_app.setStyleSheet(f"color: {PALETTE['accent']}; font-size: 20px; font-weight: 900; letter-spacing: 1px;")
        left_col.addWidget(title_app)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(280)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: transparent; }}")
        self.settings = SettingsPanel(cfg)
        scroll.setWidget(self.settings)
        left_col.addWidget(scroll)
        main_layout.addLayout(left_col)

        # Middle: Camera and Controls
        center_col = QVBoxLayout()
        center_col.setSpacing(15)

        # Wrapper for camera feed (Bezel monitor))
        cam_frame = QFrame()
        cam_frame.setStyleSheet(f"background-color: #000; border: 2px solid {PALETTE['border']}; border-radius: 12px;")
        cam_layout = QVBoxLayout(cam_frame)
        cam_layout.setContentsMargins(2, 2, 2, 2)
        
        self.lbl_camera = QLabel()
        self.lbl_camera.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_camera.setStyleSheet("background-color: transparent; border: none;")
        self.lbl_camera.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.lbl_camera.setMinimumSize(640, 480) 
        self._show_placeholder()
        cam_layout.addWidget(self.lbl_camera)
        
        center_col.addWidget(cam_frame, stretch=1)

        # Progress Buffer
        buf_row = QHBoxLayout()
        lbl_buf = QLabel("Data Buffer:")
        lbl_buf.setStyleSheet("font-weight: bold; color: #9CA3AF;")
        buf_row.addWidget(lbl_buf)
        
        self.progress_buf = QProgressBar()
        self.progress_buf.setRange(0, 100)
        self.progress_buf.setValue(0)
        buf_row.addWidget(self.progress_buf, stretch=1)
        
        self.lbl_buf_pct = QLabel("0%")
        self.lbl_buf_pct.setStyleSheet(f"color: {PALETTE['accent2']}; font-size:12px; font-weight: bold; min-width:35px;")
        buf_row.addWidget(self.lbl_buf_pct)
        center_col.addLayout(buf_row)

        # Modern Control Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        
        self.btn_start = QPushButton("▶ START")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.setFixedHeight(42)
        
        self.btn_stop  = QPushButton("■ STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setFixedHeight(42)
        self.btn_stop.setEnabled(False)
        
        self.btn_reset = QPushButton("↺ RESET")
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setFixedHeight(42)
        self.btn_reset.setEnabled(False)
        
        self.btn_save  = QPushButton("💾 EXPORT")
        self.btn_save.setObjectName("btn_save")
        self.btn_save.setFixedHeight(42)
        self.btn_save.setEnabled(False)

        self.btn_justify = QPushButton("📊 JUSTIFY CHROM")
        self.btn_justify.setObjectName("btn_justify")
        self.btn_justify.setFixedHeight(42)
        self.btn_justify.setEnabled(False)

        for b in (self.btn_start, self.btn_stop, self.btn_reset, self.btn_save, self.btn_justify):
            btn_row.addWidget(b)
            
        center_col.addLayout(btn_row)
        main_layout.addLayout(center_col, stretch=3)

        # Right: BPM Cards
        right_col = QVBoxLayout()
        right_col.setSpacing(15)

        title_hr = QLabel("REAL-TIME METRICS")
        title_hr.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 13px; font-weight: 800; letter-spacing: 1px;")
        right_col.addWidget(title_hr)

        self.bpm_cards: dict[str, BpmCard] = {}
        for mode in self.MODES:
            card = BpmCard(mode)
            self.bpm_cards[mode] = card
            right_col.addWidget(card)

        right_col.addStretch()

        info_card = QFrame()
        info_card.setStyleSheet(f"background-color: {PALETTE['panel']}; border-radius: 8px; padding: 10px;")
        info_layout = QVBoxLayout(info_card)
        info = QLabel("💡 <b>ROI Modes Info:</b><br><br><b>A</b> — Forehead Only<br><b>B</b> — Cheeks (L+R Avg)<br><b>C</b> — Combined (All)")
        info.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size:12px; line-height: 1.4;")
        info.setWordWrap(True)
        info_layout.addWidget(info)
        right_col.addWidget(info_card)

        main_layout.addLayout(right_col, stretch=1)

        self.statusBar().showMessage("Ready. System initialized successfully.")

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_justify.clicked.connect(self._on_justify)

        if not _MODULES_OK:
            self.statusBar().showMessage(f"⚠  Import error — check project structure. ({_IMPORT_ERR})")
            self.btn_start.setEnabled(False)

    def _show_placeholder(self):
        ph = np.zeros((480, 640, 3), dtype=np.uint8)
        # Warna placeholder dimodernisasi
        cv2.putText(ph, "CAMERA OFFLINE", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 100), 2)
        cv2.putText(ph, "Press START to initialize feed", (200, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 140), 1)
        self._display_frame(ph)

    def _on_start(self):
        if not _MODULES_OK:
            self.statusBar().showMessage("Cannot start: module import failed.")
            return
        cfg = self.settings.get_config()
        self._worker = MonitorWorker(cfg)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.bpm_updated.connect(self._on_bpm)
        self._worker.status_changed.connect(self.statusBar().showMessage)
        self._worker.buffer_changed.connect(self._on_buffer)
        self._worker.save_done.connect(self._on_save_done)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.settings.setEnabled(False)
        self.btn_justify.setEnabled(True)

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self.btn_stop.setEnabled(False)

    def _on_reset(self):
        if self._worker:
            self._worker.request_reset()

    def _on_save(self):
        if self._worker:
            self._worker.request_save()

    def _on_justify(self):
        if not self._worker: return
        raw_data = self._worker.get_raw_signals_snapshot('C_combined')
        
        if not raw_data or len(raw_data['R']) < 100: 
            self.statusBar().showMessage("⚠ Data belum cukup! Tunggu beberapa detik lagi (minimal ~5 detik).")
            return
            
        self.statusBar().showMessage("Sedang memproses Justifikasi CHROM (Mungkin memakan waktu ~5 detik)...")
        self.btn_justify.setEnabled(False)
        
        cfg = self.settings.get_config()
        self._justify_worker = JustifyWorker(cfg, raw_data)
        self._justify_worker.finished_plot.connect(self._on_justify_done)
        self._justify_worker.error_occurred.connect(self._on_justify_error)
        self._justify_worker.start()
    
    def _on_justify_done(self, path, results):
        self.btn_justify.setEnabled(True)
        self.statusBar().showMessage(f"✔ Justifikasi CHROM tersimpan di: {path}")
        try:
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        except Exception as e:
            print(f"[UI] Gagal membuka image viewer otomatis: {e}")
        
    def _on_justify_error(self, err):
        self.btn_justify.setEnabled(True)
        self.statusBar().showMessage(f"⚠ Error Justifikasi: {err}")

    def _on_worker_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.settings.setEnabled(True)
        self.btn_justify.setEnabled(False)
        self._show_placeholder()
        self._worker = None

    def _on_frame(self, frame: np.ndarray):
        self._display_frame(frame)

    def _on_bpm(self, bpm_data: dict):
        for mode, card in self.bpm_cards.items():
            card.update_bpm(bpm_data.get(mode, {}))

    def _on_buffer(self, n: int, buf_max: int):
        pct = int(min(n / max(buf_max, 1), 1.0) * 100)
        self.progress_buf.setValue(pct)
        self.lbl_buf_pct.setText(f"{pct}%")

    def _on_save_done(self, out_dir: str):
        self.statusBar().showMessage(f"✔  Saved to '{out_dir}/'  — {time.strftime('%H:%M:%S')}")

    def _on_error(self, msg: str):
        self.statusBar().showMessage(f"⚠  {msg}")
        self._on_worker_finished()

    def _display_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        
        lbl_size = self.lbl_camera.size()
        scaled_pix = pix.scaled(lbl_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.lbl_camera.setPixmap(scaled_pix)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        elif key == Qt.Key.Key_R and self.btn_reset.isEnabled():
            self._on_reset()
        elif key == Qt.Key.Key_S and self.btn_save.isEnabled():
            self._on_save()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("section_title")
    return lbl


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("rPPG Medical Monitor")
    app.setStyle("Fusion")

    # Dark palette base (override fallback if stylesheet fails)
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,       QColor(11, 15, 25))
    pal.setColor(QPalette.ColorRole.WindowText,   QColor(243, 244, 246))
    pal.setColor(QPalette.ColorRole.Base,         QColor(26, 33, 51))
    pal.setColor(QPalette.ColorRole.AlternateBase,QColor(19, 24, 38))
    pal.setColor(QPalette.ColorRole.Text,         QColor(243, 244, 246))
    pal.setColor(QPalette.ColorRole.Button,       QColor(26, 33, 51))
    pal.setColor(QPalette.ColorRole.ButtonText,   QColor(243, 244, 246))
    pal.setColor(QPalette.ColorRole.Highlight,    QColor(59, 130, 246))
    app.setPalette(pal)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()