"""Main application window: sidebar + tab area + worker connection"""

import pandas as pd

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QTabWidget, QScrollArea, QSplitter,
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QFileDialog, QProgressBar,
    QSizePolicy, QStatusBar,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from .theme   import PAL
from .widgets import (
    MatplotlibCanvas, BgCentralWidget, SidebarSection, LogWidget, MetricsBadge,
)
from ..config import (
    DEFAULT_IMAGE_NAMES, IMG_SHORT,
    DEFAULT_THRESHOLD_MAP, DEFAULT_PERCENTILE_MAP,
)
from ..worker import ProcessingWorker
from ..viz    import (
    create_histogram_figure, create_clahe_figure, create_segmentation_figure,
    create_stepwise_bar_figure, create_stepwise_grid_figure, create_timing_figure,
    create_summary_figure, create_runtime_figure,
    create_cross_step_figure, create_cross_timing_figure,
)

try:
    import cupy as cp
    cp.zeros(1)
    _GPU = True
except Exception:
    _GPU = False


class NucleiSegApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬  MoNuSeg Nuclei Segmentation  ·  v2.1")
        self.setMinimumSize(1500, 900)
        self.resize(1720, 980)

        self._worker      = None
        self._results     = []
        self._step_all    = {}
        self._timing_all  = {}
        self._canvases    = {}
        self._badges      = {}

        self._central = BgCentralWidget()
        self.setCentralWidget(self._central)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        main_hl = QHBoxLayout(self._central)
        main_hl.setContentsMargins(0, 0, 0, 0)
        main_hl.setSpacing(0)

        # Sidebar
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(330)
        sidebar_scroll.setStyleSheet(
            "QScrollArea {"
            "  background: rgba(4, 10, 22, 210);"
            "  border: none;"
            f"  border-right: 1px solid {PAL['border']};"
            "}")
        sidebar_scroll.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        sidebar_inner = QWidget()
        sidebar_inner.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        sidebar_inner.setStyleSheet("background: transparent;")
        sidebar_scroll.setWidget(sidebar_inner)
        self._sidebar_layout = QVBoxLayout(sidebar_inner)
        self._sidebar_layout.setContentsMargins(8, 8, 8, 12)
        self._sidebar_layout.setSpacing(10)

        self._build_sidebar()
        main_hl.addWidget(sidebar_scroll)

        # Tab area
        self._tabs = QTabWidget()
        self._tabs.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._tabs.setStyleSheet(
            "QTabWidget { margin: 0; background: transparent; }"
            "QTabWidget::pane {"
            "  background: rgba(5, 13, 30, 200);"
            f"  border: 1px solid {PAL['border']};"
            "  border-radius: 0 6px 6px 6px;"
            "}")
        self._build_main_tabs()
        main_hl.addWidget(self._tabs, 1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._prog = QProgressBar()
        self._prog.setFixedWidth(180)
        self._prog.setVisible(False)
        self._status.addPermanentWidget(self._prog)
        self._set_status("Ready — configure dataset path and click ▶ Run Processing")

        self._try_load_bg()

    # Sidebar
    def _build_sidebar(self):
        sl = self._sidebar_layout

        logo = QLabel("🔬  NucleiSeg GUI")
        logo.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        logo.setStyleSheet(
            f"color:{PAL['accent4']}; font-size:16px; font-weight:800;"
            f"background: rgba(5, 15, 35, 160);"
            f"padding:10px 4px 6px 4px; border-bottom:2px solid {PAL['accent1']};"
            f"border-radius: 4px;")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(logo)

        sub = QLabel("MoNuSeg 2018  ·  Pipeline v2.1")
        sub.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        sub.setStyleSheet(
            f"color:{PAL['text2']}; font-size:11px; padding:4px 0 6px 0;"
            f"background: transparent;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sl.addWidget(sub)

        # Dataset section
        sec_ds = SidebarSection("📁  Dataset")
        self._le_base = QLineEdit("MoNuSeg2018")
        self._le_base.setPlaceholderText("Path to MoNuSeg2018 folder…")
        btn_browse = QPushButton("Browse"); btn_browse.setObjectName("btnBrowse")
        btn_browse.setFixedWidth(68)
        btn_browse.clicked.connect(self._browse_base_dir)
        row = QWidget(); rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(4)
        rl.addWidget(self._le_base, 1); rl.addWidget(btn_browse)
        sec_ds.add_widget(row)

        self._le_bg = QLineEdit("background.jpg")
        self._le_bg.setPlaceholderText("background.jpg / .png")
        btn_bg = QPushButton("Browse"); btn_bg.setObjectName("btnBrowse")
        btn_bg.setFixedWidth(68)
        btn_bg.clicked.connect(self._browse_bg_image)
        row2 = QWidget(); rl2 = QHBoxLayout(row2)
        rl2.setContentsMargins(0, 0, 0, 0); rl2.setSpacing(4)
        rl2.addWidget(self._le_bg, 1); rl2.addWidget(btn_bg)
        lbl_bg = QLabel("Background image:")
        lbl_bg.setStyleSheet(f"color:{PAL['text2']}; font-size:11px;")
        sec_ds.add_widget(lbl_bg)
        sec_ds.add_widget(row2)
        sl.addWidget(sec_ds)

        # Stain extraction
        sec_st = SidebarSection("🧪  Stain Extraction")
        self._cb_stain = QComboBox()
        self._cb_stain.addItems(["rgb2hed", "macenko", "manual"])
        self._cb_stain.setToolTip(
            "rgb2hed: uses skimage (Ruifrok-Johnston matrix)\n"
            "macenko: SVD-based adaptive per-image estimation\n"
            "manual:  explicit R-J matrix in NumPy")
        sec_st.add_row("Mode:", self._cb_stain)
        sl.addWidget(sec_st)

        # Per-image threshold strategy
        sec_thr = SidebarSection("🎯  Threshold Strategy")
        self._thr_tabs = QTabWidget()
        self._thr_cb   = {}
        self._thr_pct  = {}
        for name in DEFAULT_IMAGE_NAMES:
            short = IMG_SHORT[name]
            tab_w = QWidget()
            tl    = QFormLayout(tab_w)
            tl.setContentsMargins(4, 6, 4, 6); tl.setSpacing(6)

            cb = QComboBox()
            cb.addItems(["percentile", "otsu", "multi_otsu", "auto"])
            cb.setCurrentText(DEFAULT_THRESHOLD_MAP.get(name, "percentile"))
            self._thr_cb[name] = cb
            tl.addRow("Strategy:", cb)

            sp = QDoubleSpinBox()
            sp.setRange(1.0, 99.9); sp.setSingleStep(0.5); sp.setDecimals(1)
            sp.setValue(DEFAULT_PERCENTILE_MAP.get(name, 68.0))
            sp.setToolTip("Percentile value (used when strategy='percentile' or 'auto')")
            self._thr_pct[name] = sp
            tl.addRow("Percentile:", sp)

            def _make_toggle(sp_=sp, cb_=cb):
                def toggle(text):
                    sp_.setEnabled(text in ("percentile", "auto"))
                return toggle
            cb.currentTextChanged.connect(_make_toggle())
            cb.currentTextChanged.emit(cb.currentText())

            self._thr_tabs.addTab(tab_w, short)

        self._thr_tabs.setStyleSheet(
            "QTabBar::tab{padding:3px 8px;font-size:10px;}"
            "QTabWidget::pane{border:none;}")
        self._thr_tabs.setFixedHeight(90)
        sec_thr.add_widget(self._thr_tabs)
        sl.addWidget(sec_thr)

        # CLAHE
        sec_cl = SidebarSection("📡  CLAHE Enhancement")
        self._ck_clahe = QCheckBox("Enable CLAHE")
        self._ck_clahe.setChecked(True)
        sec_cl.add_widget(self._ck_clahe)
        self._sp_clip = QDoubleSpinBox()
        self._sp_clip.setRange(0.1, 20.0); self._sp_clip.setSingleStep(0.1)
        self._sp_clip.setDecimals(1); self._sp_clip.setValue(1.0)
        sec_cl.add_row("Clip limit:", self._sp_clip)
        tile_w = QWidget(); tile_hl = QHBoxLayout(tile_w)
        tile_hl.setContentsMargins(0, 0, 0, 0); tile_hl.setSpacing(4)
        self._sp_tile_w = QSpinBox(); self._sp_tile_w.setRange(2, 64); self._sp_tile_w.setValue(9)
        self._sp_tile_h = QSpinBox(); self._sp_tile_h.setRange(2, 64); self._sp_tile_h.setValue(9)
        tile_hl.addWidget(self._sp_tile_w)
        tile_hl.addWidget(QLabel("×"))
        tile_hl.addWidget(self._sp_tile_h)
        sec_cl.add_row("Tile size:", tile_w)
        sl.addWidget(sec_cl)

        # Gaussian blur
        sec_gb = SidebarSection("🌀  Gaussian Blur")
        self._sp_gkern = QSpinBox()
        self._sp_gkern.setRange(1, 31); self._sp_gkern.setValue(5)
        self._sp_gkern.setSingleStep(2)
        self._sp_gkern.setToolTip("Kernel size (must be odd)")
        sec_gb.add_row("Kernel size:", self._sp_gkern)
        sl.addWidget(sec_gb)

        # Morphology
        sec_mo = SidebarSection("🔧  Morphology")
        self._sp_mks = QSpinBox(); self._sp_mks.setRange(1, 15); self._sp_mks.setValue(3)
        self._sp_oi  = QSpinBox(); self._sp_oi.setRange(0, 10);  self._sp_oi.setValue(2)
        self._sp_ci  = QSpinBox(); self._sp_ci.setRange(0, 10);  self._sp_ci.setValue(0)
        sec_mo.add_row("Kernel size:", self._sp_mks)
        sec_mo.add_row("Open iters:",  self._sp_oi)
        sec_mo.add_row("Close iters:", self._sp_ci)
        sl.addWidget(sec_mo)

        # Size filter
        sec_sz = SidebarSection("🔍  Size Filter")
        self._sp_minA = QSpinBox(); self._sp_minA.setRange(1,   5000);  self._sp_minA.setValue(20)
        self._sp_maxA = QSpinBox(); self._sp_maxA.setRange(100, 500000); self._sp_maxA.setValue(80000)
        sec_sz.add_row("Min area (px):", self._sp_minA)
        sec_sz.add_row("Max area (px):", self._sp_maxA)
        sl.addWidget(sec_sz)

        # Watershed
        sec_ws = SidebarSection("💧  Watershed")
        self._ck_ws = QCheckBox("Enable Watershed  [MOD-3]")
        self._ck_ws.setChecked(False)
        sec_ws.add_widget(self._ck_ws)
        self._sp_pmd = QSpinBox(); self._sp_pmd.setRange(1, 50); self._sp_pmd.setValue(8)
        self._sp_dtf = QDoubleSpinBox()
        self._sp_dtf.setRange(0.05, 0.95); self._sp_dtf.setSingleStep(0.01)
        self._sp_dtf.setDecimals(2); self._sp_dtf.setValue(0.28)
        sec_ws.add_row("Peak min dist:", self._sp_pmd)
        sec_ws.add_row("Dist thresh:",   self._sp_dtf)
        sl.addWidget(sec_ws)

        # Diagnostics
        sec_diag = SidebarSection("📊  Diagnostics")
        self._ck_diag = QCheckBox("Run step-wise diagnostics")
        self._ck_diag.setChecked(True)
        sec_diag.add_widget(self._ck_diag)
        self._sp_trep = QSpinBox(); self._sp_trep.setRange(1, 10); self._sp_trep.setValue(3)
        sec_diag.add_row("Timing repeats:", self._sp_trep)
        sl.addWidget(sec_diag)

        # Action buttons
        sl.addSpacing(8)
        btn_run = QPushButton("▶  Run Processing")
        btn_run.setObjectName("btnRun")
        btn_run.clicked.connect(self._on_run)
        self._btn_run = btn_run
        sl.addWidget(btn_run)

        btn_save = QPushButton("💾  Save Results CSV")
        btn_save.setObjectName("btnSave")
        btn_save.setEnabled(False)
        btn_save.clicked.connect(self._on_save_csv)
        self._btn_save = btn_save
        sl.addWidget(btn_save)

        dev_lbl = QLabel(f"  Device: {'GPU (CuPy)' if _GPU else 'CPU'}")
        dev_lbl.setStyleSheet(
            f"color:{'#00E676' if _GPU else PAL['text2']}; font-size:11px;"
            f"padding:4px 0; font-style:italic;")
        sl.addWidget(dev_lbl)
        sl.addStretch()

    # Main tabs
    def _build_main_tabs(self):
        self._tabs.clear()
        self._canvases.clear()

        # Tab 0 — Overview / Log
        ov_w = QWidget()
        ov_l = QVBoxLayout(ov_w)
        ov_l.setContentsMargins(12, 12, 12, 12); ov_l.setSpacing(10)

        hdr = QLabel(
            "<span style='font-size:20px;font-weight:800;color:#00E5FF;'>"
            "🔬 MoNuSeg Nuclei Segmentation</span><br>"
            "<span style='font-size:12px;color:#6B8FBD;'>"
            "Pipeline v2.1 · PyQt6 GUI · ITS Teknik Biomedik</span>")
        hdr.setTextFormat(Qt.TextFormat.RichText)
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(f"padding:16px; background:{PAL['bg3']}; border-radius:8px;"
                          f"border:1px solid {PAL['border']};")
        ov_l.addWidget(hdr)

        pipe_lbl = QLabel(
            "<b style='color:#42A5F5;'>Pipeline:</b><br>"
            "<span style='color:#B0C8E8;'>"
            "1. H&E Stain Extraction → 2. CLAHE Enhancement → 3. Gaussian Blur<br>"
            "→ 4. Adaptive Threshold → 5. Morphological Open → 6. Close + Hole Fill<br>"
            "→ 7. (Optional) Watershed → 8. Size Filter → IoU · Dice · Time"
            "</span>")
        pipe_lbl.setTextFormat(Qt.TextFormat.RichText)
        pipe_lbl.setStyleSheet(
            f"background:{PAL['bg3']}; padding:12px; border-radius:6px;"
            f"border:1px solid {PAL['border2']}; line-height:1.6;")
        ov_l.addWidget(pipe_lbl)

        try:
            from ..core.io import _USE_TIFFFILE
        except ImportError:
            _USE_TIFFFILE = False

        self._log_widget = LogWidget()
        self._log_widget.append_log("=== NucleiSeg GUI initialised ===")
        self._log_widget.append_log(f"  tifffile: {'✓' if _USE_TIFFFILE else '✗ (cv2 fallback)'}")
        self._log_widget.append_log(f"  GPU (CuPy): {'✓' if _GPU else '✗'}")
        self._log_widget.append_log("\nSet dataset path in sidebar and click ▶ Run Processing.")
        ov_l.addWidget(self._log_widget, 1)

        self._prog_bar_ov = QProgressBar()
        self._prog_bar_ov.setRange(0, 4); self._prog_bar_ov.setValue(0)
        self._prog_bar_ov.setFixedHeight(6)
        ov_l.addWidget(self._prog_bar_ov)
        self._tabs.addTab(ov_w, "⚙  Overview")

        # Tabs 1-5 — Per-image analysis
        tab_defs = [
            ("📊  Histograms",     "hist"),
            ("📡  CLAHE",          "clahe"),
            ("🧫  Segmentation",   "seg"),
            ("📈  Stage Analysis", "step"),
            ("⏱  Timing",         "timing"),
        ]
        for tab_label, key in tab_defs:
            outer_w = QWidget()
            outer_l = QVBoxLayout(outer_w)
            outer_l.setContentsMargins(6, 6, 6, 6)
            sub_tabs = QTabWidget()

            for name in DEFAULT_IMAGE_NAMES:
                short     = IMG_SHORT[name]
                tab_inner = QWidget()
                ti_l      = QVBoxLayout(tab_inner)
                ti_l.setContentsMargins(4, 4, 4, 4); ti_l.setSpacing(4)

                img_hdr   = QWidget()
                img_hdr_l = QHBoxLayout(img_hdr)
                img_hdr_l.setContentsMargins(6, 2, 6, 2)
                name_lbl  = QLabel(f"📌  {name}")
                name_lbl.setObjectName("imageTitle")
                img_hdr_l.addWidget(name_lbl)
                img_hdr_l.addStretch()

                if key == "seg":
                    badge = MetricsBadge()
                    self._badges[name] = badge
                    img_hdr_l.addWidget(badge)
                ti_l.addWidget(img_hdr)

                if key == "step":
                    ss = QTabWidget()
                    ss.setStyleSheet("QTabBar::tab{padding:3px 8px;font-size:11px;}")
                    for sub_label, sub_key in [
                        ("📈 Bar Chart", f"step_bar_{name}"),
                        ("🗂  Grid View", f"step_grid_{name}"),
                    ]:
                        cv = MatplotlibCanvas()
                        cv.show_placeholder(
                            f"Run processing to see {sub_label.split()[1]} for\n{short}")
                        self._canvases[sub_key] = cv
                        ss.addTab(cv, sub_label)
                    ti_l.addWidget(ss, 1)
                else:
                    cv = MatplotlibCanvas()
                    cv.show_placeholder(
                        f"Run processing to see {tab_label.strip()} for\n{short}")
                    ckey = f"{key}_{name}"
                    self._canvases[ckey] = cv
                    ti_l.addWidget(cv, 1)

                sub_tabs.addTab(tab_inner, short)

            if key in ("step", "timing"):
                cross_w = QWidget()
                cross_l = QVBoxLayout(cross_w)
                cross_l.setContentsMargins(4, 4, 4, 4)
                cv_cross = MatplotlibCanvas()
                cv_cross.show_placeholder("Run processing to see cross-image comparison")
                ckey = f"{key}_cross"
                self._canvases[ckey] = cv_cross
                cross_l.addWidget(cv_cross)
                sub_tabs.addTab(cross_w, "🔀  All Images")

            outer_l.addWidget(sub_tabs)
            self._tabs.addTab(outer_w, tab_label)

        # Tab 6 — Results Summary
        sum_w   = QWidget()
        sum_l   = QVBoxLayout(sum_w)
        sum_l.setContentsMargins(6, 6, 6, 6)
        sum_tabs = QTabWidget()

        for slbl, skey in [
            ("📊  Metrics",  "summary_metrics"),
            ("⏱  Runtimes", "summary_runtime"),
            ("📋  Table",    "summary_table"),
        ]:
            sw    = QWidget()
            sl_in = QVBoxLayout(sw)
            sl_in.setContentsMargins(4, 4, 4, 4)
            if skey != "summary_table":
                cv = MatplotlibCanvas()
                cv.show_placeholder("Run processing to view summary")
                self._canvases[skey] = cv
                sl_in.addWidget(cv)
            else:
                self._results_table = QTextEdit()
                self._results_table.setReadOnly(True)
                self._results_table.setStyleSheet(
                    f"background:{PAL['bg0']}; color:{PAL['text0']};"
                    f"font-family:monospace; font-size:12px;"
                    f"border:1px solid {PAL['border2']};")
                self._results_table.setText("Run processing to view results table.")
                sl_in.addWidget(self._results_table)
            sum_tabs.addTab(sw, slbl)

        sum_l.addWidget(sum_tabs)
        self._tabs.addTab(sum_w, "🏆  Summary")

    # Helpers
    def _set_status(self, msg: str):
        self._status.showMessage(msg)

    def _browse_base_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select MoNuSeg2018 Directory")
        if d:
            self._le_base.setText(d)

    def _browse_bg_image(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Background Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if f:
            self._le_bg.setText(f)
            self._try_load_bg()

    def _try_load_bg(self):
        from pathlib import Path
        path = self._le_bg.text() if hasattr(self, "_le_bg") else "background.jpg"
        for candidate in [path, Path(__file__).parent.parent / path]:
            p = Path(candidate)
            if p.exists():
                px = QPixmap(str(p))
                if not px.isNull():
                    self._central.set_bg_pixmap(px)
                    self._set_status(f"Background image loaded: {p.name}")
                    return
        self._central.set_bg_pixmap(None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._central.update()

    def _get_config(self) -> dict:
        ksize = self._sp_gkern.value()
        if ksize % 2 == 0:
            ksize += 1

        threshold_map  = {n: self._thr_cb[n].currentText()  for n in DEFAULT_IMAGE_NAMES}
        percentile_map = {n: self._thr_pct[n].value()       for n in DEFAULT_IMAGE_NAMES}

        params = {
            "use_clahe":        self._ck_clahe.isChecked(),
            "clahe_clip_limit": self._sp_clip.value(),
            "clahe_tile_size":  (self._sp_tile_w.value(), self._sp_tile_h.value()),
            "gaussian_ksize":   (ksize, ksize),
            "morph_kernel_size": self._sp_mks.value(),
            "open_iterations":  self._sp_oi.value(),
            "close_iterations": self._sp_ci.value(),
            "min_area_px":      self._sp_minA.value(),
            "max_area_px":      self._sp_maxA.value(),
            "peak_min_dist":    self._sp_pmd.value(),
            "dist_thresh_frac": self._sp_dtf.value(),
        }
        return {
            "base_dir":        self._le_base.text(),
            "image_names":     DEFAULT_IMAGE_NAMES,
            "stain_mode":      self._cb_stain.currentText(),
            "use_watershed":   self._ck_ws.isChecked(),
            "params":          params,
            "threshold_map":   threshold_map,
            "percentile_map":  percentile_map,
            "run_diagnostics": self._ck_diag.isChecked(),
            "timing_repeats":  self._sp_trep.value(),
        }

    # Run/worker slots
    def _on_run(self):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._btn_run.setText("▶  Run Processing")
            self._btn_run.setEnabled(True)
            self._set_status("Processing aborted.")
            return

        cfg = self._get_config()
        self._results    = []
        self._step_all   = {}
        self._timing_all = {}
        self._log_widget.clear()
        self._log_widget.append_log(
            f"▶ Starting processing\n"
            f"  Base dir  : {cfg['base_dir']}\n"
            f"  Stain     : {cfg['stain_mode']}\n"
            f"  Watershed : {cfg['use_watershed']}\n"
            f"  CLAHE     : {cfg['params']['use_clahe']}  "
            f"clip={cfg['params']['clahe_clip_limit']}\n"
            f"  Kernel    : {cfg['params']['gaussian_ksize']}\n"
            f"  Morph     : open={cfg['params']['open_iterations']}  "
            f"close={cfg['params']['close_iterations']}")
        self._prog_bar_ov.setValue(0)
        self._prog.setRange(0, len(DEFAULT_IMAGE_NAMES))
        self._prog.setValue(0)
        self._prog.setVisible(True)
        self._btn_run.setText("⏹  Stop")
        self._btn_save.setEnabled(False)
        self._set_status("Processing…")

        self._worker = ProcessingWorker(cfg)
        self._worker.log.connect(self._log_widget.append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.hist_ready.connect(self._on_hist_ready)
        self._worker.clahe_ready.connect(self._on_clahe_ready)
        self._worker.seg_ready.connect(self._on_seg_ready)
        self._worker.step_ready.connect(self._on_step_ready)
        self._worker.timing_ready.connect(self._on_timing_ready)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished_.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int, name: str):
        self._prog.setValue(current)
        self._prog_bar_ov.setValue(current)
        self._set_status(
            f"Processing [{current}/{total}]: {IMG_SHORT.get(name, name)}")

    def _on_hist_ready(self, name: str, image_rgb, H_u8):
        try:
            fig = create_histogram_figure(image_rgb, H_u8, IMG_SHORT.get(name, name))
            key = f"hist_{name}"
            if key in self._canvases:
                self._canvases[key].update_figure(fig)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Hist figure error: {e}")

    def _on_clahe_ready(self, name: str, H_u8_raw, params: dict):
        try:
            fig = create_clahe_figure(H_u8_raw, params, IMG_SHORT.get(name, name))
            key = f"clahe_{name}"
            if key in self._canvases:
                self._canvases[key].update_figure(fig)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ CLAHE figure error: {e}")

    def _on_seg_ready(self, name: str, image_rgb, gt_mask, pred_mask, metrics: dict):
        try:
            fig = create_segmentation_figure(
                image_rgb, gt_mask, pred_mask, metrics, IMG_SHORT.get(name, name))
            key = f"seg_{name}"
            if key in self._canvases:
                self._canvases[key].update_figure(fig)
            if name in self._badges:
                self._badges[name].update_metrics(metrics)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Seg figure error: {e}")

    def _on_step_ready(self, name: str, step_data: dict):
        short = IMG_SHORT.get(name, name)
        try:
            fig_bar = create_stepwise_bar_figure(
                step_data["stages"], step_data["ious"], step_data["dices"], short)
            bkey = f"step_bar_{name}"
            if bkey in self._canvases:
                self._canvases[bkey].update_figure(fig_bar)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Step bar figure error: {e}")

        try:
            gkey = f"step_grid_{name}"
            if gkey in self._canvases:
                if "image_rgb" in step_data and "gt_mask" in step_data:
                    fig_grid = create_stepwise_grid_figure(
                        step_data["image_rgb"], step_data["gt_mask"],
                        step_data["masks"], step_data["stages"],
                        step_data["ious"], step_data["dices"], short)
                    self._canvases[gkey].update_figure(fig_grid)
                else:
                    self._canvases[gkey].show_placeholder(
                        "Grid needs image data — will be available after re-run")
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Step grid figure error: {e}")

        self._step_all[name] = step_data
        if len(self._step_all) == len(DEFAULT_IMAGE_NAMES):
            self._render_cross_step()

    def _on_timing_ready(self, name: str, timing_data: dict):
        try:
            fig = create_timing_figure(
                timing_data["stages"], timing_data["times_ms"],
                timing_data["total_ms"], timing_data["n_components"],
                IMG_SHORT.get(name, name))
            key = f"timing_{name}"
            if key in self._canvases:
                self._canvases[key].update_figure(fig)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Timing figure error: {e}")

        self._timing_all[name] = timing_data
        if len(self._timing_all) == len(DEFAULT_IMAGE_NAMES):
            self._render_cross_timing()

    def _render_cross_step(self):
        try:
            fig = create_cross_step_figure(self._step_all)
            if "step_cross" in self._canvases:
                self._canvases["step_cross"].update_figure(fig)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Cross-step figure error: {e}")

    def _render_cross_timing(self):
        try:
            fig = create_cross_timing_figure(self._timing_all)
            if "timing_cross" in self._canvases:
                self._canvases["timing_cross"].update_figure(fig)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Cross-timing figure error: {e}")

    def _on_all_done(self, results: list):
        self._results = results
        if not results:
            return

        df = pd.DataFrame(results)

        try:
            fig_sum = create_summary_figure(df)
            if "summary_metrics" in self._canvases:
                self._canvases["summary_metrics"].update_figure(fig_sum)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Summary figure error: {e}")

        try:
            fig_rt = create_runtime_figure(df)
            if "summary_runtime" in self._canvases:
                self._canvases["summary_runtime"].update_figure(fig_rt)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Runtime figure error: {e}")

        try:
            cols = ["Image", "IoU", "Dice", "Precision", "Recall",
                    "Running Time", "Threshold", "Pct"]
            cols = [c for c in cols if c in df.columns]
            table_str = (
                f"{'='*72}\n  RESULTS SUMMARY — MoNuSeg 2018  ·  v2.1\n{'='*72}\n\n"
                + df[cols].to_string(index=False)
                + f"\n\n{'─'*72}\n"
                + f"  Mean IoU   : {df['IoU'].mean():.4f}  (±{df['IoU'].std():.4f})\n"
                + f"  Mean Dice  : {df['Dice'].mean():.4f}  (±{df['Dice'].std():.4f})\n"
                + f"  Mean Time  : {df['Running Time'].mean():.4f} s\n"
                + f"{'='*72}\n"
            )
            self._results_table.setText(table_str)
        except Exception as e:
            self._log_widget.append_log(f"  ✗ Table error: {e}")

        self._btn_save.setEnabled(True)
        self._tabs.setCurrentIndex(6)

    def _on_error(self, msg: str):
        self._log_widget.append_log(f"\n⚠  ERROR: {msg}")
        self._set_status(f"Error: {msg}")

    def _on_finished(self):
        self._btn_run.setText("▶  Run Processing")
        self._btn_run.setEnabled(True)
        self._prog.setVisible(False)
        self._prog_bar_ov.setValue(4)
        self._set_status(
            f"✓  Processing complete — {len(self._results)} image(s) processed")

    def _on_save_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results CSV", "results_v2.csv", "CSV Files (*.csv)")
        if path:
            pd.DataFrame(self._results).to_csv(path, index=False)
            self._set_status(f"Results saved → {path}")
            self._log_widget.append_log(f"\n  ✓ CSV saved: {path}")

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait(2000)
        import matplotlib.pyplot as plt
        plt.close("all")
        super().closeEvent(event)
