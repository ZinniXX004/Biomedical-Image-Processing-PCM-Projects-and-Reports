import os
import sys
import datetime
import numpy as np
from skimage import data as skdata

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
                             QSlider, QComboBox, QPushButton, QStatusBar, QFileDialog,
                             QScrollArea, QFrame, QTabWidget, QApplication)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor

# Local Imports
from config import (BG_DARK, BG_PANEL, BG_CARD, TEXT_COL, SUBTEXT, GRID_COL,
                    STAGE_COLORS, METHOD_COLORS, ENH_COLORS, EDGE_CMAP, KERNELS, KERNEL_INFO)

# Import our manual math functions
from processing import (compute_pipeline, compute_all_enhancements, 
                        compute_enhancement_metrics, manual_rgb2gray)

from components import ImagePanel, AnalysisPanel, BackgroundWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬 Interactive Edge Detection Viewer — PCM Assignment (Modular UI)")
        self.resize(1750, 1000)
        self._apply_dark_palette()

        self.gray_img = None
        self.image_name = "unknown"
        self._last_result = None

        self.enh_clahe_clip = 0.03
        self.cs_low = 2.0
        self.cs_high = 98.0

        self._load_default_image()

        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._do_update)

        self._autosave_timer = QTimer()
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._do_autosave)

        self.panels = {}
        self.analysis_panels = {}

        self._build_ui()
        self._update()

    def _apply_dark_palette(self):
        p = QPalette()
        for role, hex_col in[
            (QPalette.ColorRole.Window, BG_DARK),
            (QPalette.ColorRole.WindowText, TEXT_COL),
            (QPalette.ColorRole.Base, BG_PANEL),
            (QPalette.ColorRole.AlternateBase, BG_CARD),
            (QPalette.ColorRole.Text, TEXT_COL),
            (QPalette.ColorRole.Button, BG_CARD),
            (QPalette.ColorRole.ButtonText, TEXT_COL),
        ]:
            p.setColor(role, QColor(hex_col))
        QApplication.instance().setPalette(p)

    def _load_default_image(self):
        astro = skdata.astronaut()
        h, w = min(astro.shape[0], 300), min(astro.shape[1], 300)
        # Use our manual math RGB to Grayscale
        self.gray_img = manual_rgb2gray(astro[:h, :w])
        self.image_name = "Astronaut (skimage built-in)"

    def _build_ui(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.custom_bg_path = os.path.join(base_dir, "Stultifera Navis.jpg")   
        
        self.bg_widget = BackgroundWidget(
            image_path=self.custom_bg_path,
            opacity=0.45,   
            dimness=0.60    
        )
        self.setCentralWidget(self.bg_widget)
        
        root = QHBoxLayout(self.bg_widget)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(16)

        self.left_panel = self._build_controls()
        root.addWidget(self.left_panel)
        
        self.tabs = self._build_tabbed_view()
        root.addWidget(self.tabs, 1)

        self.status = QStatusBar()
        self.status.setStyleSheet(
            f"QStatusBar{{ background: rgba(23, 26, 33, 0.90); color:{SUBTEXT}; font-size:11px; padding:6px; font-family:'Segoe UI'; border-top:1px solid {GRID_COL}; }}"
        )
        self.setStatusBar(self.status)
    
    def _create_card(self, title: str, accent_color: str) -> tuple[QWidget, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("ModernCard")
        card.setStyleSheet(f"#ModernCard {{ background-color: {BG_CARD}; border: 1px solid {GRID_COL}; border-radius: 10px; }}")
        outer_layout = QVBoxLayout(card)
        outer_layout.setContentsMargins(14, 14, 14, 14)
        outer_layout.setSpacing(10)

        if title:
            header = QLabel(title)
            header.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            header.setStyleSheet(f"color: {accent_color}; border: none; background: transparent;")
            outer_layout.addWidget(header)

            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet(f"background-color: {GRID_COL}; border: none; max-height: 1px;")
            outer_layout.addWidget(line)

        content_layout = QVBoxLayout()
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(0, 4, 0, 0)
        outer_layout.addLayout(content_layout)

        return card, content_layout

    def _build_controls(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(340)
        scroll.setStyleSheet(
            f"QScrollArea{{ background: transparent; border:none; }}"
            f"QScrollBar:vertical{{ background: transparent; width:8px; border:none; }}"
            f"QScrollBar::handle:vertical{{ background:{GRID_COL}; border-radius:4px; }}"
        )

        w = QWidget()
        w.setStyleSheet(f"background: transparent;")
        layout = QVBoxLayout(w)
        layout.setSpacing(14)
        layout.setContentsMargins(0, 0, 10, 0)

        title = QLabel("⚙️ Edge Detection Controls")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{TEXT_COL}; padding-bottom:4px;")
        layout.addWidget(title)

        card_m, cm_layout = self._create_card("🔍 Detection Method (Tab 4)", METHOD_COLORS['Prewitt'])
        self.method_combo = QComboBox()
        self.method_combo.addItems(list(KERNELS.keys()))
        self.method_combo.setStyleSheet(
            f"QComboBox {{ background:{BG_PANEL}; color:{TEXT_COL}; padding:8px 12px; font-size:12px; font-family:'Segoe UI'; border:1px solid {GRID_COL}; border-radius:6px; }}"
            f"QComboBox::drop-down {{ border:none; }}"
            f"QComboBox QAbstractItemView {{ background:{BG_PANEL}; color:{TEXT_COL}; selection-background-color:{STAGE_COLORS['gradient']}; selection-color:{BG_DARK}; outline:none; }}"
        )
        self.method_combo.currentTextChanged.connect(self._update)
        cm_layout.addWidget(self.method_combo)
        layout.addWidget(card_m)

        card_enh, cenh_layout = self._create_card("✨ Enhancement Method", STAGE_COLORS['enhancement'])
        self.enh_combo = QComboBox()
        self.enh_combo.addItems(["CLAHE", "Histogram Equalization", "Contrast Stretching", "None (Original)"])
        self.enh_combo.setStyleSheet(self.method_combo.styleSheet())
        self.enh_combo.currentTextChanged.connect(self._update)
        cenh_layout.addWidget(self.enh_combo)
        
        hint_enh = QLabel("Selects which method is passed down the pipeline.")
        hint_enh.setStyleSheet(f"color:{SUBTEXT}; font-size:11px; font-family:'Segoe UI';")
        hint_enh.setWordWrap(True)
        cenh_layout.addWidget(hint_enh)
        layout.addWidget(card_enh)

        card_s, cs_layout = self._create_card("Gaussian σ — Restoration (Tab 3)", STAGE_COLORS['restoration'])
        self.sigma_lbl = QLabel("σ = 0.80")
        self.sigma_lbl.setStyleSheet(f"color:{STAGE_COLORS['restoration']}; font-weight:bold; font-size:12px; font-family:'Segoe UI';")
        self.sigma_slider = QSlider(Qt.Orientation.Horizontal)
        self.sigma_slider.setRange(0, 50)
        self.sigma_slider.setValue(8)
        self.sigma_slider.setStyleSheet(
            f"QSlider::groove:horizontal{{ background:{GRID_COL}; height:6px; border-radius:3px; }}"
            f"QSlider::handle:horizontal{{ background:{STAGE_COLORS['restoration']}; width:16px; height:16px; margin:-5px 0; border-radius:8px; }}"
            f"QSlider::sub-page:horizontal{{ background:{STAGE_COLORS['restoration']}; border-radius:3px; }}"
        )
        self.sigma_slider.valueChanged.connect(self._sigma_changed)
        hint_s = QLabel("0 = no denoising  |  50 = strong blur")
        hint_s.setStyleSheet(f"color:{SUBTEXT}; font-size:11px; font-family:'Segoe UI';")
        cs_layout.addWidget(self.sigma_lbl)
        cs_layout.addWidget(self.sigma_slider)
        cs_layout.addWidget(hint_s)
        layout.addWidget(card_s)

        card_t, ct_layout = self._create_card("Edge Threshold — Results (Tab 5)", STAGE_COLORS['results'])
        self.thr_lbl = QLabel("threshold = 0.12")
        self.thr_lbl.setStyleSheet(f"color:{STAGE_COLORS['results']}; font-weight:bold; font-size:12px; font-family:'Segoe UI';")
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(1, 60)
        self.thr_slider.setValue(12)
        self.thr_slider.setStyleSheet(
            f"QSlider::groove:horizontal{{ background:{GRID_COL}; height:6px; border-radius:3px; }}"
            f"QSlider::handle:horizontal{{ background:{STAGE_COLORS['results']}; width:16px; height:16px; margin:-5px 0; border-radius:8px; }}"
            f"QSlider::sub-page:horizontal{{ background:{STAGE_COLORS['results']}; border-radius:3px; }}"
        )
        self.thr_slider.valueChanged.connect(self._thr_changed)
        hint_t = QLabel("lower = more edges  |  higher = fewer")
        hint_t.setStyleSheet(f"color:{SUBTEXT}; font-size:11px; font-family:'Segoe UI';")
        ct_layout.addWidget(self.thr_lbl)
        ct_layout.addWidget(self.thr_slider)
        ct_layout.addWidget(hint_t)
        layout.addWidget(card_t)

        btn_load = QPushButton("📂  Load Facial Image")
        btn_load.setMinimumHeight(44)
        btn_load.setStyleSheet(
            f"QPushButton{{ background:{BG_CARD}; color:{TEXT_COL}; border:1px solid {GRID_COL}; border-radius:8px; font-weight:bold; font-size:13px; font-family:'Segoe UI'; }}"
            f"QPushButton:hover{{ background:{BG_PANEL}; border-color:{STAGE_COLORS['preprocessing']}; }}"
            f"QPushButton:pressed{{ background:{STAGE_COLORS['preprocessing']}; color:{BG_DARK}; }}"
        )
        btn_load.clicked.connect(self._load_image)
        layout.addWidget(btn_load)

        btn_save = QPushButton("💾  Save All Panels Now")
        btn_save.setMinimumHeight(44)
        btn_save.setStyleSheet(
            f"QPushButton{{ background:{BG_CARD}; color:{STAGE_COLORS['results']}; border:1px solid {STAGE_COLORS['results']}50; border-radius:8px; font-weight:bold; font-size:13px; font-family:'Segoe UI'; }}"
            f"QPushButton:hover{{ background:{BG_PANEL}; border-color:{STAGE_COLORS['results']}; }}"
            f"QPushButton:pressed{{ background:{STAGE_COLORS['results']}; color:{BG_DARK}; }}"
        )
        btn_save.clicked.connect(self._save_all_now)
        layout.addWidget(btn_save)

        card_stats, cstats_layout = self._create_card("📊 Live Metrics", TEXT_COL)
        self.stats_lbl = QLabel("Run detection to see metrics")
        self.stats_lbl.setFont(QFont("Consolas", 11))
        self.stats_lbl.setStyleSheet(f"color:{SUBTEXT}; line-height: 1.5;")
        self.stats_lbl.setWordWrap(True)
        cstats_layout.addWidget(self.stats_lbl)
        layout.addWidget(card_stats)

        card_info, cinfo_layout = self._create_card("📐 Kernel Info", SUBTEXT)
        self.info_lbl = QLabel("")
        self.info_lbl.setFont(QFont("Consolas", 10))
        self.info_lbl.setStyleSheet(f"color:{SUBTEXT};")
        self.info_lbl.setWordWrap(True)
        cinfo_layout.addWidget(self.info_lbl)
        layout.addWidget(card_info)

        self.autosave_lbl = QLabel("💾 Auto-save: enabled (3 s delay)")
        self.autosave_lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:11px; padding-left:4px; font-family:'Segoe UI';")
        self.autosave_lbl.setWordWrap(True)
        layout.addWidget(self.autosave_lbl)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_tabbed_view(self) -> QTabWidget:
        tab_style = (
            f"QTabWidget::pane {{ border:1px solid {GRID_COL}; border-radius: 10px; background: rgba(27, 40, 56, 0.45); padding: 4px; }}" 
            f"QTabBar::tab {{ background:{BG_DARK}; color:{SUBTEXT}; padding:10px 18px; margin-right: 4px; margin-bottom: 8px; border:1px solid {GRID_COL}; border-radius:8px; font-family:'Segoe UI'; font-size:12px; font-weight:bold; }}"
            f"QTabBar::tab:selected {{ background:{BG_CARD}; color:{TEXT_COL}; border:1px solid {STAGE_COLORS['preprocessing']}; }}"
            f"QTabBar::tab:hover {{ background:{BG_CARD}; color:{TEXT_COL}; }}"
        )
        tabs = QTabWidget()
        tabs.setStyleSheet(tab_style)
        tabs.addTab(self._build_tab_acquisition(),  " 1. Acquisition ")
        tabs.addTab(self._build_tab_enhancement(),  " 2. Image Enhancement ")
        tabs.addTab(self._build_tab_restoration(),  " 3. Restoration ")
        tabs.addTab(self._build_tab_gradient(),     " 4. Gradient and Detection ")
        tabs.addTab(self._build_tab_results(),      " 5. Results and Analysis ")
        tabs.addTab(self._build_tab_secret(),       " ✧ ")
        tabs.currentChanged.connect(self._on_tab_changed)

        return tabs

    def _build_tab_secret(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent; border: none;")
        return w

    def _on_tab_changed(self, index: int):
        if not hasattr(self, 'left_panel') or not hasattr(self, 'status') or not hasattr(self, 'bg_widget'):
            return

        if index == 5: 
            self.left_panel.hide()
            self.status.hide()
            self.bg_widget.opacity = 1.0
            self.bg_widget.dimness = 0.0
            self.bg_widget.update()
        else:
            self.left_panel.show()
            self.status.show()
            self.bg_widget.opacity = 0.45
            self.bg_widget.dimness = 0.60
            self.bg_widget.update()

    def _scroll_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Made tab scroll area transparent to reveal background
        scroll.setStyleSheet(
            f"QScrollArea{{ background: transparent; border:none; }}"
            f"QScrollBar:vertical  {{ background: transparent; width:10px; border:none; }}"
            f"QScrollBar:horizontal{{ background: transparent; height:10px; border:none; }}"
            f"QScrollBar::handle:vertical, QScrollBar::handle:horizontal"
            f"  {{ background:{GRID_COL}; border-radius:5px; }}"
        )
        content = QWidget()
        content.setStyleSheet(f"background: transparent;")
        layout = QVBoxLayout(content)
        layout.setSpacing(16)
        layout.setContentsMargins(12, 12, 12, 12)
        scroll.setWidget(content)
        return scroll, layout

    def _section_header(self, stage: str, desc: str, color: str) -> QWidget:
        w = QWidget()
        w.setFixedHeight(48)
        w.setStyleSheet(f"background:{BG_CARD}; border-radius:8px; border-left:6px solid {color};")
        l = QHBoxLayout(w)
        l.setContentsMargins(16, 4, 16, 4)
        l.setSpacing(14)

        s_lbl = QLabel(stage)
        s_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        s_lbl.setStyleSheet(f"color:{color}; background:transparent; border:none;")
        
        sep = QLabel("│")
        sep.setStyleSheet(f"color:{GRID_COL}; background:transparent; border:none; font-size:16px;")
        
        d_lbl = QLabel(desc)
        d_lbl.setFont(QFont("Segoe UI", 11))
        d_lbl.setStyleSheet(f"color:{SUBTEXT}; background:transparent; border:none;")

        l.addWidget(s_lbl); l.addWidget(sep); l.addWidget(d_lbl)
        l.addStretch()
        return w

    def _panel_row(self, widgets: list) -> QWidget:
        row = QWidget()
        row.setStyleSheet(f"background:transparent;")
        rl = QHBoxLayout(row)
        rl.setSpacing(12)
        rl.setContentsMargins(0, 0, 0, 0)
        for wgt in widgets:
            rl.addWidget(wgt)
        return row

    def _slider_card(self, label_text, range_lo, range_hi, default_val, unit="", color="#89B4FA"):
        card, layout = self._create_card(label_text, color)
        val_lbl = QLabel(f"{default_val}{unit}")
        val_lbl.setStyleSheet(f"color:{color}; font-weight:bold; font-size:13px; font-family:'Segoe UI';")
        
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(range_lo, range_hi)
        sld.setValue(default_val)
        sld.setStyleSheet(
            f"QSlider::groove:horizontal{{ background:{BG_DARK}; height:6px; border-radius:3px; }}"
            f"QSlider::handle:horizontal{{ background:{color}; width:16px; height:16px; margin:-5px 0; border-radius:8px; }}"
            f"QSlider::sub-page:horizontal{{ background:{color}; border-radius:3px; }}"
        )
        layout.addWidget(val_lbl)
        layout.addWidget(sld)
        return card, sld, val_lbl

    def _build_tab_acquisition(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()
        layout.addWidget(self._section_header("1. ACQUISITION", "Original Grayscale → np.clip[0,1] → Acquired Image", STAGE_COLORS["preprocessing"]))

        self.panels["acq_orig"] = ImagePanel("Original Gray", "As loaded / rgb2gray", "preprocessing")
        self.panels["acq"] = ImagePanel("1. Acquired", "np.clip(gray, 0, 1)", "preprocessing")
        layout.addWidget(self._panel_row([self.panels["acq_orig"], self.panels["acq"]]))

        layout.addWidget(self._section_header("  ANALYSIS", "Histogram and Ogive (CDF) Original vs Acquired", STAGE_COLORS["preprocessing"]))

        self.analysis_panels["acq_orig_ha"] = AnalysisPanel("Histogram + Ogive", "Original Gray", "preprocessing", figsize=(4.5, 3.2))
        self.analysis_panels["acq_ha"] = AnalysisPanel("Histogram + Ogive", "Acquired Image", "preprocessing", figsize=(4.5, 3.2))
        self.analysis_panels["acq_compare_ha"] = AnalysisPanel("Comparison Overlay", "Original vs Acquired", "preprocessing", figsize=(5.0, 3.2))
        
        layout.addWidget(self._panel_row([self.analysis_panels["acq_orig_ha"], self.analysis_panels["acq_ha"], self.analysis_panels["acq_compare_ha"]]))
        layout.addStretch()
        return scroll

    def _build_tab_enhancement(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()
        layout.addWidget(self._section_header("2. IMAGE ENHANCEMENT CONTROLS", "Adjust parameters for each method (CS, HE, CLAHE)", STAGE_COLORS["enhancement"]))

        ctrl_row = QWidget()
        ctrl_row.setStyleSheet(f"background:transparent;")
        ctrl_l = QHBoxLayout(ctrl_row)
        ctrl_l.setSpacing(14)
        ctrl_l.setContentsMargins(0, 0, 0, 0)

        card_cl, self.enh_clahe_sld, self.enh_clahe_lbl = self._slider_card("CLAHE clip_limit (×0.01)", 1, 10, 3, unit=" → 0.03", color=ENH_COLORS["CLAHE"])
        self.enh_clahe_sld.valueChanged.connect(self._enh_clahe_changed)
        ctrl_l.addWidget(card_cl)

        card_cslo, self.enh_cs_low_sld, self.enh_cs_low_lbl = self._slider_card("CS p_low (%)", 0, 15, 2, unit="%", color=ENH_COLORS["CS"])
        self.enh_cs_low_sld.valueChanged.connect(self._enh_cs_low_changed)
        ctrl_l.addWidget(card_cslo)

        card_cshi, self.enh_cs_high_sld, self.enh_cs_high_lbl = self._slider_card("CS p_high (%)", 80, 100, 98, unit="%", color=ENH_COLORS["CS"])
        self.enh_cs_high_sld.valueChanged.connect(self._enh_cs_high_changed)
        ctrl_l.addWidget(card_cshi)

        info_lbl = QLabel("<b>CS</b> = Contrast Stretching<br><b>HE</b> = Histogram Equalization<br><b>CLAHE</b> = Adaptive HE<br><br><i>Modifying these will auto-update the main pipeline if selected.</i>")
        info_lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:12px; font-family:'Segoe UI'; padding: 12px; background-color:{BG_CARD}; border-radius:10px; border:1px solid {GRID_COL};")
        info_lbl.setWordWrap(True)
        ctrl_l.addWidget(info_lbl, 1)
        layout.addWidget(ctrl_row)

        layout.addWidget(self._section_header("  ENHANCED IMAGES", "CS | Histogram Equalization | CLAHE side-by-side comparison", STAGE_COLORS["enhancement"]))

        self.panels["enh_cs"]    = ImagePanel("Contrast Stretching", "Percentile linear", "enhancement")
        self.panels["enh_he"]    = ImagePanel("Histogram Equalization", "Global HE", "enhancement")
        self.panels["enh_clahe"] = ImagePanel("CLAHE", "Adaptive HE", "enhancement")
        layout.addWidget(self._panel_row([self.panels["enh_cs"], self.panels["enh_he"], self.panels["enh_clahe"]]))

        layout.addWidget(self._section_header("  HISTOGRAM and OGIVE PER METHOD", "Pixel intensity distribution + cumulative distribution function", STAGE_COLORS["enhancement"]))

        self.analysis_panels["enh_cs_ha"]    = AnalysisPanel("CS Hist + Ogive", "Contrast Stretching", "enhancement", figsize=(4.0, 3.2))
        self.analysis_panels["enh_he_ha"]    = AnalysisPanel("HE Hist + Ogive", "Histogram Equalization", "enhancement", figsize=(4.0, 3.2))
        self.analysis_panels["enh_clahe_ha"] = AnalysisPanel("CLAHE Hist + Ogive", "Adaptive HE", "enhancement", figsize=(4.0, 3.2))
        layout.addWidget(self._panel_row([self.analysis_panels["enh_cs_ha"], self.analysis_panels["enh_he_ha"], self.analysis_panels["enh_clahe_ha"]]))

        layout.addWidget(self._section_header("  PERFORMANCE EVALUATION", "RMSE · PSNR · SSIM · Shannon Entropy (reference = acquired image)", STAGE_COLORS["enhancement"]))

        self.analysis_panels["enh_perf"] = AnalysisPanel("Performance Evaluation", "RMSE / PSNR / SSIM / Shannon Entropy", "enhancement", figsize=(10.0, 4.0))
        self.analysis_panels["enh_perf"].setMinimumHeight(350)
        layout.addWidget(self.analysis_panels["enh_perf"])

        layout.addStretch()
        return scroll

    def _build_tab_restoration(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()
        layout.addWidget(self._section_header("3. RESTORATION and MORPHO PRE-PROC", "Gaussian Denoising (σ slider in left panel) applied on selected Enhancement → Morphological Closing", STAGE_COLORS["restoration"]))

        self.panels["denoised"]   = ImagePanel("3. Restoration",  "Gaussian σ-controlled", "restoration")
        self.panels["morpho_pre"] = ImagePanel("4. Morpho Pre-proc", "Structural Closing disk(1)", "restoration")
        layout.addWidget(self._panel_row([self.panels["denoised"], self.panels["morpho_pre"]]))

        layout.addStretch()
        return scroll

    def _build_tab_gradient(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()
        layout.addWidget(self._section_header("4. GRADIENT and EDGE DETECTION", "Kernel Convolution → Gradient Gx & Gy → Magnitude √(Gx²+Gy²) → Direction arctan2", STAGE_COLORS["gradient"]))

        self.panels["gx"]        = ImagePanel("4. Gradient Gx",   "Horizontal Edges", "gradient")
        self.panels["gy"]        = ImagePanel("5. Gradient Gy",   "Vertical Edges",   "gradient")
        self.panels["magnitude"] = ImagePanel("6. Edge Magnitude","√(Gx²+Gy²) Norm.", "gradient")
        self.panels["direction"] = ImagePanel("7. Direction Map", "arctan2(Gy, Gx)",  "gradient")
        layout.addWidget(self._panel_row([self.panels["gx"], self.panels["gy"], self.panels["magnitude"], self.panels["direction"]]))

        layout.addStretch()
        return scroll

    def _build_tab_results(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()
        layout.addWidget(self._section_header("5. RESULTS and ANALYSIS", "Thresholding → Morpho Thinning (Skeleton) → Magnitude Distribution → Row Profile", STAGE_COLORS["results"]))

        self.panels["binary"]      = ImagePanel("5. Thresholded",    "Binary Edge Map",      "results")
        self.panels["morpho_post"] = ImagePanel("6. Morpho Post-proc","Skeletonize / Thin",  "results")
        layout.addWidget(self._panel_row([self.panels["binary"], self.panels["morpho_post"]]))

        layout.addWidget(self._section_header("  MAGNITUDE ANALYSIS", "Histogram (with threshold marker) · Histogram + Ogive (CDF) · Row Profile", STAGE_COLORS["results"]))

        self.panels["histogram"] = ImagePanel("11. Magnitude Hist.", "Distribution + threshold marker", "results")
        self.panels["profile"]   = ImagePanel("12. Row Profile",     "Intensity Cross-section",         "results")
        self.analysis_panels["result_mag_ha"] = AnalysisPanel("Magnitude Hist + Ogive", "Edge magnitude CDF for threshold selection", "results", figsize=(4.5, 3.2))

        layout.addWidget(self._panel_row([self.panels["histogram"], self.analysis_panels["result_mag_ha"], self.panels["profile"]]))

        layout.addStretch()
        return scroll

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Facial Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)")
        if not path:
            return
        try:
            if HAS_CV2:
                bgr = cv2.imread(path)
                if bgr is not None:
                    rgb = cv2.cvtColor(cv2.resize(bgr, (300, 300)), cv2.COLOR_BGR2RGB)
                    # Use manual mathematical RGB to Grayscale
                    self.gray_img = manual_rgb2gray(rgb)
            else:
                from PIL import Image as PILImage
                pil = PILImage.open(path).convert("RGB").resize((300, 300))
                # Use manual mathematical RGB to Grayscale
                self.gray_img = manual_rgb2gray(np.array(pil))
            
            self.image_name = os.path.basename(path)
            self._update()
            self.status.showMessage(f"Loaded: {path}", 5000)
        except Exception as e:
            self.status.showMessage(f"Error loading image: {e}", 5000)

    def _sigma_changed(self, val):
        sigma = val / 10.0
        self.sigma_lbl.setText(f"σ = {sigma:.2f}")
        self._update()

    def _thr_changed(self, val):
        thr = val / 100.0
        self.thr_lbl.setText(f"threshold = {thr:.2f}")
        self._update()

    def _enh_clahe_changed(self, val: int):
        self.enh_clahe_clip = val / 100.0
        self.enh_clahe_lbl.setText(f"clip_limit = {self.enh_clahe_clip:.2f}")
        self._update()

    def _enh_cs_low_changed(self, val: int):
        self.cs_low = float(val)
        self.enh_cs_low_lbl.setText(f"p_low = {val}%")
        self._update()

    def _enh_cs_high_changed(self, val: int):
        self.cs_high = float(val)
        self.enh_cs_high_lbl.setText(f"p_high = {val}%")
        self._update()

    def _update(self):
        self._debounce.start(180)

    def _do_update(self):
        if self.gray_img is None:
            return
        method = self.method_combo.currentText()
        sigma  = self.sigma_slider.value() / 10.0
        thr    = self.thr_slider.value()  / 100.0

        enh_text = self.enh_combo.currentText()
        if "CLAHE" in enh_text: enh_m = "CLAHE"
        elif "Histogram" in enh_text: enh_m = "HE"
        elif "Contrast" in enh_text: enh_m = "CS"
        else: enh_m = "None"

        res = compute_pipeline(
            self.gray_img, method, sigma, thr,
            enh_method=enh_m, clahe_clip=self.enh_clahe_clip,
            cs_low=self.cs_low, cs_high=self.cs_high
        )
        self._last_result = res
        m_color = METHOD_COLORS.get(method, TEXT_COL)

        self.panels["acq_orig"].show_image(self.gray_img, "gray")
        self.panels["acq"].show_image(res["acq"], "gray")
        self.analysis_panels["acq_orig_ha"].show_hist_ogive(self.gray_img, label="Original Gray", color="#89B4FA")
        self.analysis_panels["acq_ha"].show_hist_ogive(res["acq"], label="Acquired Image", color="#FAB387")
        self.analysis_panels["acq_compare_ha"].show_comparison_hist_ogive({"Original": self.gray_img, "Acquired": res["acq"]}, title="Original vs Acquired")

        self.panels["denoised"].show_image(res["denoised"],   "gray")
        self.panels["morpho_pre"].show_image(res["morpho_pre"], "gray")

        self.panels["gx"].show_image(res["gx"], "RdBu_r")
        self.panels["gy"].show_image(res["gy"], "PRGn_r")
        self.panels["magnitude"].show_image(res["magnitude"], EDGE_CMAP, colorbar=True)
        self.panels["direction"].show_image(res["direction"], "hsv", colorbar=True)

        self.panels["binary"].show_image(res["binary"], "gray")
        self.panels["morpho_post"].show_image(res["morpho_post"], "gray")
        self.panels["histogram"].show_histogram(res["magnitude"], thr, m_color)
        self.panels["profile"].show_profile(res["magnitude"], res["binary"], thr, m_color, row=self.gray_img.shape[0] // 2)
        self.analysis_panels["result_mag_ha"].show_hist_ogive(res["magnitude"], label=f"Edge Magnitude — {method}", color=m_color)

        self.stats_lbl.setText(
            f"Edge Method:  {method}\n"
            f"Enhancement:  {enh_m}\n"
            f"Runtime:      {res['elapsed']:.2f} ms\n"
            f"Edge Density: {res['density']:.4f}\n"
            f"Mean Mag:     {res['mean_mag']:.4f}\n"
            f"σ (sigma):    {sigma:.2f}\n"
            f"Threshold:    {thr:.2f}\n"
            f"Image:        {self.gray_img.shape[1]}×{self.gray_img.shape[0]} px\n"
            f"Source:       {self.image_name}"
        )
        self.stats_lbl.setStyleSheet(f"color:{m_color};")
        self.info_lbl.setText(KERNEL_INFO.get(method, ""))

        self.status.showMessage(f"  {method}  |  Enh={enh_m}  |  σ={sigma:.2f}  |  thr={thr:.2f}  |  density={res['density']:.4f}  |  ⏱ {res['elapsed']:.1f} ms")

        self._do_update_enhancement()
        self._autosave_timer.start(3000)

    def _do_update_enhancement(self):
        if self._last_result is None or self.gray_img is None: return
        acq = self._last_result["acq"]
        enhs = compute_all_enhancements(acq, clahe_clip=self.enh_clahe_clip, cs_low=self.cs_low, cs_high=self.cs_high)

        key_map = {"CS": "enh_cs", "HE": "enh_he", "CLAHE": "enh_clahe"}
        for mname, panel_key in key_map.items():
            self.panels[panel_key].show_image(enhs[mname], "gray")

        ha_map = {"CS": ("enh_cs_ha", "Contrast Stretching"), "HE": ("enh_he_ha", "Histogram Equalization"), "CLAHE": ("enh_clahe_ha", "CLAHE (Adaptive)")}
        for mname, (ap_key, lbl) in ha_map.items():
            self.analysis_panels[ap_key].show_hist_ogive(enhs[mname], label=lbl, color=ENH_COLORS[mname])

        metrics = {mname: compute_enhancement_metrics(acq, enhs[mname]) for mname in enhs}
        self.analysis_panels["enh_perf"].show_performance_eval(metrics, ENH_COLORS)

    def _do_autosave(self):
        self._save_panels(auto=True)

    def _save_all_now(self):
        self._save_panels(auto=False)

    def _save_panels(self, auto: bool = False):
        if self._last_result is None: return
        method = self.method_combo.currentText()
        sigma  = self.sigma_slider.value() / 10.0
        thr    = self.thr_slider.value()  / 100.0
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subdir = os.path.join(self.output_dir, ts)
        os.makedirs(subdir, exist_ok=True)

        suffix = f"{method.replace(' ','_')}_s{sigma:.1f}_t{thr:.2f}.png"
        ok = 0
        for key, panel in {**self.panels, **self.analysis_panels}.items():
            fname = f"{key}_{suffix}"
            if panel.save_to(os.path.join(subdir, fname)): ok += 1

        total = len(self.panels) + len(self.analysis_panels)
        label = "Auto-saved" if auto else "Saved"
        msg   = f"💾 {label} {ok}/{total} panels → {subdir}"
        self.autosave_lbl.setText(msg)
        self.autosave_lbl.setStyleSheet(f"color:{STAGE_COLORS['results']}; font-size:11px; padding-left:4px; font-family:'Segoe UI';")
        if not auto: self.status.showMessage(msg, 6000)