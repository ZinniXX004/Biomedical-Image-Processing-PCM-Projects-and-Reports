import sys
import os
import datetime
import numpy as np
from skimage import data as skdata

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QSlider, QComboBox, QPushButton,
        QStatusBar, QFileDialog, QScrollArea, QFrame, QSizePolicy,
        QTabWidget
    )
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont, QPalette, QColor
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    sys.exit(1)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# IMPORT MODULAR COMPONENTS
from config import (
    BG_DARK, BG_PANEL, BG_CARD, TEXT_COL, SUBTEXT, GRID_COL,
    STAGE_COLORS, METHOD_COLORS, ENH_COLORS, KERNELS, KERNEL_INFO, 
    EDGE_CMAP, CANNY_DIR_CMAP
)
from gui.components import ImagePanel, AnalysisPanel, BackgroundWidget

# IMPORT CORE LOGIC
from core.math_ops import manual_rgb2gray
from core.enhancement import compute_all_enhancements, compute_enhancement_metrics
from core.edge_detection import canny_scratch, canny_library
from core.sharpening import laplacian_sharpening, unsharp_masking
from core.feature_extraction import (
    harris_corner_scratch, harris_corner_library,
    hough_line_scratch, hough_line_library,
    hough_circle_scratch, hough_circle_library
)
from core.pipeline import compute_pipeline, compute_all_methods, compute_all_w11_methods


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬 Interactive Edge Detection Viewer (Modern UI)")
        self.resize(1750, 1000)
        self._apply_dark_palette()

        self.gray_img     = None
        self.image_name   = "unknown"
        self._last_result = None

        self.enh_clahe_clip = 0.03
        self.cs_low         = 2.0
        self.cs_high        = 98.0

        # Canny + Sharpening state
        self.canny_sigma  = 1.0
        self.canny_t_lo   = 0.05
        self.canny_t_hi   = 0.15
        self.lap_weight   = 1.0
        self.lap_kernel   = "H4"
        self.usm_a        = 0.7
        self.usm_sigma    = 1.0
        # cached results (None = needs computation)
        self._last_canny_scratch  = None
        self._last_canny_lib      = None
        self._last_sharpening_lap = None
        self._last_sharpening_usm = None
        self._last_runtimes       = None
        self._runtime_dirty       = True

        # Assignment 3 variables
        self.harris_alpha = 0.05
        self.harris_sigma = 1.0
        self.harris_thr   = 0.05
        self.hline_theta  = 180
        self.hline_thr    = 50
        self.hcirc_radius = 15
        self.hcirc_thr    = 0.45
        self._w11_runtime_dirty = True

        self._load_default_image()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, "..", "corner_line_circle_detection_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._do_update)

        self._autosave_timer = QTimer()
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._do_autosave)

        # Debounce timers for the two new tabs (longer delay = less UI lag)
        self._canny_debounce = QTimer()
        self._canny_debounce.setSingleShot(True)
        self._canny_debounce.timeout.connect(self._do_update_canny)

        self._sharp_debounce = QTimer()
        self._sharp_debounce.setSingleShot(True)
        self._sharp_debounce.timeout.connect(self._do_update_sharpening)

        self._harris_debounce = QTimer(); self._harris_debounce.setSingleShot(True); self._harris_debounce.timeout.connect(self._do_update_harris)
        self._hline_debounce = QTimer(); self._hline_debounce.setSingleShot(True); self._hline_debounce.timeout.connect(self._do_update_hline)
        self._hcirc_debounce = QTimer(); self._hcirc_debounce.setSingleShot(True); self._hcirc_debounce.timeout.connect(self._do_update_hcirc)

        self.panels          = {}
        self.analysis_panels = {}

        self._build_ui()
        self._update()
        # Trigger initial Canny + Sharpening after the pipeline settles
        self._canny_debounce.start(800)
        self._sharp_debounce.start(800)
        self._harris_debounce.start(800)
        self._hline_debounce.start(800)
        self._hcirc_debounce.start(800)

    def _apply_dark_palette(self):
        p = QPalette()
        for role, hex_col in[
            (QPalette.ColorRole.Window,        BG_DARK),
            (QPalette.ColorRole.WindowText,    TEXT_COL),
            (QPalette.ColorRole.Base,          BG_PANEL),
            (QPalette.ColorRole.AlternateBase, BG_CARD),
            (QPalette.ColorRole.Text,          TEXT_COL),
            (QPalette.ColorRole.Button,        BG_CARD),
            (QPalette.ColorRole.ButtonText,    TEXT_COL),
        ]:
            p.setColor(role, QColor(hex_col))
        QApplication.instance().setPalette(p)

    def _load_default_image(self):
        astro = skdata.astronaut()
        h = min(astro.shape[0], 300)
        w = min(astro.shape[1], 300)
        self.gray_img   = manual_rgb2gray(astro[:h, :w])
        self.image_name = "Astronaut (skimage built-in)"

    def _build_ui(self):
        # CUSTOM BACKGROUND SETUP
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.custom_bg_path = os.path.join(base_dir, '..', "Stultifera Navis.jpg")   
        
        self.bg_widget = BackgroundWidget(
            image_path=self.custom_bg_path,
            opacity=0.45,   # Image visibility (0.0 to 1.0)
            dimness=0.60    # Dark overlay intensity (0.0 to 1.0)
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
    
    # Custom UI Components
    def _create_card(self, title: str, accent_color: str) -> tuple[QWidget, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("ModernCard")
        card.setStyleSheet(f"""
            #ModernCard {{
                background-color: {BG_CARD};
                border: 1px solid {GRID_COL};
                border-radius: 10px;
            }}
        """)
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

        card_s, cs_layout = self._create_card("Gaussian σ Restoration (Tab 3)", STAGE_COLORS['restoration'])
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

        card_t, ct_layout = self._create_card("Edge Threshold Results (Tab 5)", STAGE_COLORS['results'])
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
        main_tabs = QTabWidget()
        main_tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: none; background: transparent; }}"
            f"QTabBar::tab {{ background: {BG_DARK}; color: {SUBTEXT}; padding: 12px 20px; font-size: 14px; font-weight: bold; margin-right: 5px; border-radius: 8px; }}"
            f"QTabBar::tab:selected {{ background: {BG_PANEL}; color: {TEXT_COL}; }}"
            f"QTabBar::tab:hover {{ background: {BG_CARD}; color: #ffffff; }}"
        )
        sub_tab_style = (
            f"QTabWidget::pane {{ border:1px solid {GRID_COL}; border-radius: 10px; background: rgba(27, 40, 56, 0.45); padding: 4px; }}" 
            f"QTabBar::tab {{ background:{BG_DARK}; color:{SUBTEXT}; padding:10px 18px; margin-right: 4px; margin-bottom: 8px; border:1px solid {GRID_COL}; border-radius:8px; font-family:'Segoe UI'; font-size:12px; font-weight:bold; }}"
            f"QTabBar::tab:selected {{ background:{BG_CARD}; color:{TEXT_COL}; border:1px solid {STAGE_COLORS['preprocessing']}; }}"
            f"QTabBar::tab:hover {{ background:{BG_CARD}; color:{TEXT_COL}; }}"
        )

        tabs_a1 = QTabWidget(); tabs_a1.setStyleSheet(sub_tab_style)
        tabs_a1.addTab(self._build_tab_acquisition(), " 1. Acquisition ")
        tabs_a1.addTab(self._build_tab_enhancement(), " 2. Enhancement ")
        tabs_a1.addTab(self._build_tab_restoration(), " 3. Restoration ")
        tabs_a1.addTab(self._build_tab_gradient(), " 4. Gradient ")
        tabs_a1.addTab(self._build_tab_results(), " 5. Results ")

        tabs_a2 = QTabWidget(); tabs_a2.setStyleSheet(sub_tab_style.replace(STAGE_COLORS['preprocessing'], STAGE_COLORS['canny']))
        tabs_a2.addTab(self._build_tab_canny(), " 6. Canny Edge ")
        tabs_a2.addTab(self._build_tab_sharpening(), " 7. Sharpening ")
        tabs_a2.addTab(self._build_tab_runtime_comparison(), " 8. A2 Benchmark (Runtime and Comparison) ")

        tabs_a3 = QTabWidget(); tabs_a3.setStyleSheet(sub_tab_style.replace(STAGE_COLORS['preprocessing'], STAGE_COLORS['harris']))
        tabs_a3.addTab(self._build_tab_harris(), " 9. Harris Corner ")
        tabs_a3.addTab(self._build_tab_hough_line(), " 10. Hough Line ")
        tabs_a3.addTab(self._build_tab_hough_circle(), " 11. Hough Circle ")
        tabs_a3.addTab(self._build_tab_w11_comparison(), " 12. A3 Benchmark (Runtime and Comparison) ")

        main_tabs.addTab(tabs_a1, " 📚 ASSIGNMENT 1 ")
        main_tabs.addTab(tabs_a2, " 📚 ASSIGNMENT 2 ")
        main_tabs.addTab(tabs_a3, " 📚 ASSIGNMENT 3 ")
        main_tabs.addTab(self._build_tab_secret(), " ✧ ")

        main_tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs_a2 = tabs_a2; self.tabs_a2.currentChanged.connect(self._on_sub_tab_changed)
        self.tabs_a3 = tabs_a3; self.tabs_a3.currentChanged.connect(self._on_sub_tab_changed)
        return main_tabs

    def _build_tab_secret(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent; border: none;")
        return w

    def _on_tab_changed(self, index: int):
        if not hasattr(self, 'left_panel'): return
        if index == 3:   
            self.left_panel.hide(); self.status.hide(); self.bg_widget.opacity = 1.0; self.bg_widget.dimness = 0.0; self.bg_widget.update()
        else:
            self.left_panel.show(); self.status.show(); self.bg_widget.opacity = 0.45; self.bg_widget.dimness = 0.60; self.bg_widget.update()
            if index == 1 and hasattr(self, 'tabs_a2') and self.tabs_a2.currentIndex() == 2 and getattr(self, '_runtime_dirty', False): QTimer.singleShot(150, self._do_update_runtime)
            if index == 2 and hasattr(self, 'tabs_a3') and self.tabs_a3.currentIndex() == 3 and getattr(self, '_w11_runtime_dirty', False): QTimer.singleShot(150, self._do_update_w11_runtime)
    
    def _on_sub_tab_changed(self, index: int):
        if self.tabs.currentIndex() == 1 and index == 2 and getattr(self, '_runtime_dirty', False): QTimer.singleShot(150, self._do_update_runtime)
        if self.tabs.currentIndex() == 2 and index == 3 and getattr(self, '_w11_runtime_dirty', False): QTimer.singleShot(150, self._do_update_w11_runtime)

    def _scroll_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea{{ background: transparent; border:none; }}"
            f"QScrollBar:vertical  {{ background: transparent; width:10px; border:none; }}"
            f"QScrollBar:horizontal{{ background: transparent; height:10px; border:none; }}"
            f"QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ background:{GRID_COL}; border-radius:5px; }}"
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
        w.setStyleSheet(
            f"background:{BG_CARD}; border-radius:8px;"
            f"border-left:6px solid {color};"
        )
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
            f"QSlider::groove:horizontal{{ background:{GRID_COL}; height:6px; border-radius:3px; }}"
            f"QSlider::handle:horizontal{{ background:{color}; width:16px; height:16px; margin:-5px 0; border-radius:8px; }}"
            f"QSlider::sub-page:horizontal{{ background:{color}; border-radius:3px; }}"
        )
        layout.addWidget(val_lbl)
        layout.addWidget(sld)
        return card, sld, val_lbl

    def _build_tab_acquisition(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "1. ACQUISITION", "Original Grayscale → np.clip [0,1] → Acquired Image", STAGE_COLORS["preprocessing"]))

        self.panels["acq_orig"] = ImagePanel("Original Gray", "As loaded / rgb2gray", "preprocessing")
        self.panels["acq"] = ImagePanel("1. Acquired", "np.clip(gray, 0, 1)", "preprocessing")
        layout.addWidget(self._panel_row([self.panels["acq_orig"], self.panels["acq"]]))

        layout.addWidget(self._section_header(
            "  ANALYSIS", "Histogram and Ogive (CDF) Original vs Acquired", STAGE_COLORS["preprocessing"]))

        self.analysis_panels["acq_orig_ha"] = AnalysisPanel(
            "Histogram + Ogive", "Original Gray", "preprocessing", figsize=(4.5, 3.2))
        self.analysis_panels["acq_ha"] = AnalysisPanel(
            "Histogram + Ogive", "Acquired Image", "preprocessing", figsize=(4.5, 3.2))
        self.analysis_panels["acq_compare_ha"] = AnalysisPanel(
            "Comparison Overlay", "Original vs Acquired", "preprocessing", figsize=(5.0, 3.2))
        
        layout.addWidget(self._panel_row([
            self.analysis_panels["acq_orig_ha"],
            self.analysis_panels["acq_ha"],
            self.analysis_panels["acq_compare_ha"],
        ]))

        layout.addStretch()
        return scroll

    def _build_tab_enhancement(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "2. IMAGE ENHANCEMENT CONTROLS", "Adjust parameters for each method (CS, HE, CLAHE)", STAGE_COLORS["enhancement"]))

        ctrl_row = QWidget()
        ctrl_row.setStyleSheet(f"background:transparent;")
        ctrl_l = QHBoxLayout(ctrl_row)
        ctrl_l.setSpacing(14)
        ctrl_l.setContentsMargins(0, 0, 0, 0)

        card_cl, self.enh_clahe_sld, self.enh_clahe_lbl = self._slider_card(
            "CLAHE clip_limit (×0.01)", 1, 10, 3, unit=" → 0.03", color=ENH_COLORS["CLAHE"])
        self.enh_clahe_sld.valueChanged.connect(self._enh_clahe_changed)
        ctrl_l.addWidget(card_cl)

        card_cslo, self.enh_cs_low_sld, self.enh_cs_low_lbl = self._slider_card(
            "CS p_low (%)", 0, 15, 2, unit="%", color=ENH_COLORS["CS"])
        self.enh_cs_low_sld.valueChanged.connect(self._enh_cs_low_changed)
        ctrl_l.addWidget(card_cslo)

        card_cshi, self.enh_cs_high_sld, self.enh_cs_high_lbl = self._slider_card(
            "CS p_high (%)", 80, 100, 98, unit="%", color=ENH_COLORS["CS"])
        self.enh_cs_high_sld.valueChanged.connect(self._enh_cs_high_changed)
        ctrl_l.addWidget(card_cshi)

        info_lbl = QLabel(
            "<b>CS</b> = Contrast Stretching<br>"
            "<b>HE</b> = Histogram Equalization<br>"
            "<b>CLAHE</b> = Adaptive HE<br><br>"
            "<i>Modifying these will auto-update the main pipeline if selected.</i>")
        info_lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:12px; font-family:'Segoe UI'; padding: 12px; background-color:{BG_CARD}; border-radius:10px; border:1px solid {GRID_COL};")
        info_lbl.setWordWrap(True)
        ctrl_l.addWidget(info_lbl, 1)
        layout.addWidget(ctrl_row)

        layout.addWidget(self._section_header(
            "  ENHANCED IMAGES", "CS | Histogram Equalization | CLAHE side-by-side comparison", STAGE_COLORS["enhancement"]))

        self.panels["enh_cs"]    = ImagePanel("Contrast Stretching", "Percentile linear", "enhancement")
        self.panels["enh_he"]    = ImagePanel("Histogram Equalization", "Global HE", "enhancement")
        self.panels["enh_clahe"] = ImagePanel("CLAHE", "Adaptive HE", "enhancement")
        layout.addWidget(self._panel_row([self.panels["enh_cs"], self.panels["enh_he"], self.panels["enh_clahe"]]))

        layout.addWidget(self._section_header(
            "  HISTOGRAM and OGIVE PER METHOD", "Pixel intensity distribution + cumulative distribution function", STAGE_COLORS["enhancement"]))

        self.analysis_panels["enh_cs_ha"]    = AnalysisPanel("CS Hist + Ogive", "Contrast Stretching", "enhancement", figsize=(4.0, 3.2))
        self.analysis_panels["enh_he_ha"]    = AnalysisPanel("HE Hist + Ogive", "Histogram Equalization", "enhancement", figsize=(4.0, 3.2))
        self.analysis_panels["enh_clahe_ha"] = AnalysisPanel("CLAHE Hist + Ogive", "Adaptive HE", "enhancement", figsize=(4.0, 3.2))
        layout.addWidget(self._panel_row([
            self.analysis_panels["enh_cs_ha"],
            self.analysis_panels["enh_he_ha"],
            self.analysis_panels["enh_clahe_ha"],
        ]))

        layout.addWidget(self._section_header(
            "  PERFORMANCE EVALUATION", "RMSE · PSNR · SSIM · Shannon Entropy (reference = acquired image)", STAGE_COLORS["enhancement"]))

        self.analysis_panels["enh_perf"] = AnalysisPanel(
            "Performance Evaluation", "RMSE / PSNR / SSIM / Shannon Entropy", "enhancement", figsize=(10.0, 4.0))
        self.analysis_panels["enh_perf"].setMinimumHeight(350)
        layout.addWidget(self.analysis_panels["enh_perf"])

        layout.addStretch()
        return scroll

    def _build_tab_restoration(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "3. RESTORATION and MORPHO PRE-PROC", "Gaussian Denoising (σ slider in left panel) applied on selected Enhancement → Morphological Closing", STAGE_COLORS["restoration"]))

        self.panels["denoised"]   = ImagePanel("3. Restoration",  "Gaussian σ-controlled", "restoration")
        self.panels["morpho_pre"] = ImagePanel("4. Morpho Pre-proc", "Structural Closing disk(1)", "restoration")
        layout.addWidget(self._panel_row([self.panels["denoised"], self.panels["morpho_pre"]]))

        layout.addStretch()
        return scroll

    def _build_tab_gradient(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "4. GRADIENT and EDGE DETECTION", "Kernel Convolution → Gradient Gx and Gy → Magnitude √(Gx²+Gy²) → Direction arctan2", STAGE_COLORS["gradient"]))

        self.panels["gx"]        = ImagePanel("5. Gradient Gx",   "Horizontal Edges", "gradient")
        self.panels["gy"]        = ImagePanel("6. Gradient Gy",   "Vertical Edges",   "gradient")
        self.panels["magnitude"] = ImagePanel("7. Edge Magnitude","√(Gx²+Gy²) Norm.", "gradient")
        self.panels["direction"] = ImagePanel("8. Direction Map", "arctan2(Gy, Gx)",  "gradient")
        layout.addWidget(self._panel_row([
            self.panels["gx"], self.panels["gy"], self.panels["magnitude"], self.panels["direction"]
        ]))

        layout.addStretch()
        return scroll

    def _build_tab_results(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "5. RESULTS and ANALYSIS", "Thresholding → Morpho Thinning (Skeleton) → Magnitude Distribution → Row Profile", STAGE_COLORS["results"]))

        self.panels["binary"]      = ImagePanel("9. Thresholded",    "Binary Edge Map",      "results")
        self.panels["morpho_post"] = ImagePanel("10. Morpho Post-proc","Skeletonize / Thin",  "results")
        layout.addWidget(self._panel_row([self.panels["binary"], self.panels["morpho_post"]]))

        layout.addWidget(self._section_header(
            "  MAGNITUDE ANALYSIS", "Histogram (with threshold marker) · Histogram + Ogive (CDF) · Row Profile", STAGE_COLORS["results"]))

        self.panels["histogram"] = ImagePanel("11. Magnitude Hist.", "Distribution + threshold marker", "results")
        self.panels["profile"]   = ImagePanel("12. Row Profile",     "Intensity Cross-section",         "results")
        self.analysis_panels["result_mag_ha"] = AnalysisPanel("Magnitude Hist + Ogive", "Edge magnitude CDF for threshold selection", "results", figsize=(4.5, 3.2))

        layout.addWidget(self._panel_row([
            self.panels["histogram"],
            self.analysis_panels["result_mag_ha"],
            self.panels["profile"],
        ]))

        layout.addStretch()
        return scroll

    # TAB 6: CANNY EDGE DETECTION 
    def _build_tab_canny(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "6. CANNY EDGE DETECTION",
            "Manual from-scratch implementation + skimage library comparison",
            STAGE_COLORS["canny"]))

        # Inline parameter controls
        ctrl = QWidget()
        ctrl.setStyleSheet("background:transparent;")
        cl = QHBoxLayout(ctrl)
        cl.setSpacing(14); cl.setContentsMargins(0,0,0,0)

        card_s, self.canny_sigma_sld, self.canny_sigma_lbl = self._slider_card(
            "Gaussian σ (×0.1)", 1, 30, 10, unit=" → 1.0", color=STAGE_COLORS["canny"])
        self.canny_sigma_sld.valueChanged.connect(self._canny_sigma_changed)
        cl.addWidget(card_s)

        card_lo, self.canny_tlo_sld, self.canny_tlo_lbl = self._slider_card(
            "Low Threshold (×0.01)", 1, 30, 5, unit=" → 0.05", color="#FAB387")
        self.canny_tlo_sld.valueChanged.connect(self._canny_tlo_changed)
        cl.addWidget(card_lo)

        card_hi, self.canny_thi_sld, self.canny_thi_lbl = self._slider_card(
            "High Threshold (×0.01)", 5, 50, 15, unit=" → 0.15", color="#A6E3A1")
        self.canny_thi_sld.valueChanged.connect(self._canny_thi_changed)
        cl.addWidget(card_hi)

        info_c = QLabel(
            "<b>Pre-processing:</b> Gaussian smoothing → Gaussian gradient (fx, fy)<br>"
            "<b>Edge Localization:</b> Magnitude → Direction → Digitize angle → NMS<br>"
            "<b>Hysteresis:</b> Double threshold (Lo=weak, Hi=strong) → edge tracking")
        info_c.setStyleSheet(
            f"color:{SUBTEXT}; font-size:12px; font-family:'Segoe UI';"
            f" padding:12px; background:{BG_CARD}; border-radius:10px;"
            f" border:1px solid {GRID_COL};")
        info_c.setWordWrap(True)
        cl.addWidget(info_c, 1)
        layout.addWidget(ctrl)

        # Pre-processing row
        layout.addWidget(self._section_header(
            "  PRE-PROCESSING: Gaussian Smoothing + Gaussian Gradient",
            "Ī = I * H^{G,σ}  →  Īx = ∂H/∂x   Īy = ∂H/∂y",
            STAGE_COLORS["canny"]))

        self.panels["canny_smooth"] = ImagePanel("Smoothed (Ī)",     "Gaussian σ", "canny")
        self.panels["canny_fx"]     = ImagePanel("Gradient fx (Īx)", "∂H^{G,σ}/∂x", "canny")
        self.panels["canny_fy"]     = ImagePanel("Gradient fy (Īy)", "∂H^{G,σ}/∂y", "canny")
        self.panels["canny_mag"]    = ImagePanel("Magnitude (Emag)", "√(fx²+fy²)", "canny")
        layout.addWidget(self._panel_row([
            self.panels["canny_smooth"], self.panels["canny_fx"],
            self.panels["canny_fy"],     self.panels["canny_mag"],
        ]))

        # Edge Localization + Hysteresis row 
        layout.addWidget(self._section_header(
            "  EDGE LOCALIZATION + HYSTERESIS THRESHOLDING",
            "Φ(u,v)=arctan2(Iy,Ix) → Digitize → NMS → Double threshold → Hysteresis",
            STAGE_COLORS["canny"]))

        self.panels["canny_angle"]  = ImagePanel("Direction Φ(u,v)",    "arctan2(Iy,Ix) [HSV]", "canny")
        self.panels["canny_quant"]  = ImagePanel("Digitized Direction",  "0=H  1=↗  2=V  3=↘",  "canny")
        self.analysis_panels["canny_color"] = AnalysisPanel(
            "Color Direction", "R=horiz  G=diag↗  B=vert  Y=diag↘", "canny", figsize=(3.5, 3.0))
        self.panels["canny_nms"]    = ImagePanel("Non-Max Suppressed",   "Enms",                 "canny")
        self.panels["canny_thresh"] = ImagePanel("Double Threshold",     "White=strong  Grey=weak","canny")
        self.panels["canny_hys"]    = ImagePanel("Hysteresis (Final)",   "Edge map",              "canny")
        layout.addWidget(self._panel_row([
            self.panels["canny_angle"],
            self.panels["canny_quant"],
            self.analysis_panels["canny_color"],
            self.panels["canny_nms"],
            self.panels["canny_thresh"],
            self.panels["canny_hys"],
        ]))

        # Canny Library
        layout.addWidget(self._section_header(
            "  CANNY (LIBRARY: skimage.feature.canny)",
            "Same σ / lo / hi thresholds applied via optimised library implementation",
            STAGE_COLORS["canny"]))

        self.panels["canny_lib"]  = ImagePanel("Canny Library",    "skimage.feature.canny", "canny")
        self.analysis_panels["canny_compare"] = AnalysisPanel(
            "Scratch vs Library", "Edge pixel distribution comparison",
            "canny", figsize=(6.0, 3.2))
        layout.addWidget(self._panel_row([
            self.panels["canny_lib"],
            self.analysis_panels["canny_compare"],
        ]))

        layout.addStretch()
        return scroll

    # TAB 7: IMAGE SHARPENING 
    def _build_tab_sharpening(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "7. IMAGE SHARPENING",
            "Laplacian Operator  I' = I − w·(H^L * I)   ·   Unsharp Masking  I' = I + a·(I − blur)",
            STAGE_COLORS["sharpening"]))

        # Controls
        ctrl = QWidget()
        ctrl.setStyleSheet("background:transparent;")
        cl = QHBoxLayout(ctrl)
        cl.setSpacing(14); cl.setContentsMargins(0,0,0,0)

        card_w, self.lap_weight_sld, self.lap_weight_lbl = self._slider_card(
            "Laplacian weight w (×0.1)", 1, 30, 10, unit=" → 1.0", color=STAGE_COLORS["sharpening"])
        self.lap_weight_sld.valueChanged.connect(self._lap_weight_changed)
        cl.addWidget(card_w)

        # Kernel selector (QComboBox inside a card)
        kern_card, kern_layout = self._create_card("Laplacian Kernel", STAGE_COLORS["sharpening"])
        self.lap_kernel_combo = QComboBox()
        self.lap_kernel_combo.addItems(["H4  (4-conn  |  0,1,0 / 1,-4,1 / 0,1,0)",
                                        "H8  (8-conn  |  1,1,1 / 1,-8,1 / 1,1,1)",
                                        "H12 (weighted|  1,2,1 / 2,-12,2 / 1,2,1)"])
        self.lap_kernel_combo.setStyleSheet(
            f"QComboBox {{background:{BG_PANEL}; color:{TEXT_COL}; padding:8px 12px; font-size:11px;"
            f" border:1px solid {GRID_COL}; border-radius:6px; }}"
            f"QComboBox QAbstractItemView {{background:{BG_PANEL}; color:{TEXT_COL}; }}")
        self.lap_kernel_combo.currentTextChanged.connect(self._lap_kernel_changed)
        kern_layout.addWidget(self.lap_kernel_combo)
        cl.addWidget(kern_card)

        card_a, self.usm_a_sld, self.usm_a_lbl = self._slider_card(
            "USM factor a (×0.1)", 1, 30, 7, unit=" → 0.7", color="#89B4FA")
        self.usm_a_sld.valueChanged.connect(self._usm_a_changed)
        cl.addWidget(card_a)

        card_us, self.usm_sigma_sld, self.usm_sigma_lbl = self._slider_card(
            "USM Gaussian σ (×0.1)", 1, 30, 10, unit=" → 1.0", color="#89B4FA")
        self.usm_sigma_sld.valueChanged.connect(self._usm_sigma_changed)
        cl.addWidget(card_us)

        layout.addWidget(ctrl)

        # Laplacian section 
        layout.addWidget(self._section_header(
            "  LAPLACIAN SHARPENING",
            "Separable: H_x=[1,-2,1]  H_y=[[1],[-2],[1]]  →  Full 2D kernel H^L",
            STAGE_COLORS["sharpening"]))

        self.panels["sharp_orig"]     = ImagePanel("Input (Gray)",       "Original grayscale",   "sharpening")
        self.panels["sharp_lap_x"]    = ImagePanel("Laplacian X",        "|H_x * blur|",         "sharpening")
        self.panels["sharp_lap_y"]    = ImagePanel("Laplacian Y",        "|H_y * blur|",         "sharpening")
        self.panels["sharp_lap_sum"]  = ImagePanel("Lap XY Sum",         "|Lap_x + Lap_y|",      "sharpening")
        self.panels["sharp_sep"]      = ImagePanel("Sharpened (Sep.)",   "I − w·(Lap_x+Lap_y)", "sharpening")
        layout.addWidget(self._panel_row([
            self.panels["sharp_orig"],    self.panels["sharp_lap_x"],
            self.panels["sharp_lap_y"],   self.panels["sharp_lap_sum"],
            self.panels["sharp_sep"],
        ]))

        self.panels["sharp_lap_full"] = ImagePanel("Laplacian H^L",      "|H^L * blur|",         "sharpening")
        self.panels["sharp_full"]     = ImagePanel("Sharpened (Full)",   "I − w·(H^L*I)",       "sharpening")
        layout.addWidget(self._panel_row([
            self.panels["sharp_lap_full"], self.panels["sharp_full"],
        ]))

        # Unsharp Masking section 
        layout.addWidget(self._section_header(
            "  UNSHARP MASKING (USM)",
            "M = I − blur(I, σ)   →   I' = I + a · M",
            STAGE_COLORS["sharpening"]))

        self.panels["sharp_usm_blur"]   = ImagePanel("USM Blurred",     "Gaussian blur", "sharpening")
        self.panels["sharp_usm_mask"]   = ImagePanel("USM Mask  M",     "I − blur",      "sharpening")
        self.panels["sharp_usm_result"] = ImagePanel("USM Sharpened I'","I + a·M",       "sharpening")
        layout.addWidget(self._panel_row([
            self.panels["sharp_orig"],           # reuse the same panel reference
            self.panels["sharp_usm_blur"],
            self.panels["sharp_usm_mask"],
            self.panels["sharp_usm_result"],
        ]))

        layout.addWidget(self._section_header("  INTENSITY PROFILE COMPARISON", "Comparison of pixel intensity across an image row", STAGE_COLORS["sharpening"]))
        self.analysis_panels["sharp_profile"] = AnalysisPanel("Intensity Profile", "Original vs Laplacian vs USM", "sharpening", figsize=(12.0, 3.5))
        self.analysis_panels["sharp_profile"].setMinimumHeight(280)
        layout.addWidget(self.analysis_panels["sharp_profile"])

        layout.addStretch()
        return scroll

    # TAB 8: RUNTIME AND COMPARISON
    def _build_tab_runtime_comparison(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()

        layout.addWidget(self._section_header(
            "8. RUNTIME and METHOD COMPARISON",
            "Run all methods → measure elapsed time → compare edge outputs side-by-side",
            STAGE_COLORS["comparison"]))

        # Button to force re-run
        btn_run = QPushButton("▶  Run All Methods Now")
        btn_run.setMinimumHeight(44)
        btn_run.setStyleSheet(
            f"QPushButton {{ background:{BG_CARD}; color:{STAGE_COLORS['comparison']};"
            f" border:1px solid {STAGE_COLORS['comparison']}60; border-radius:8px;"
            f" font-weight:bold; font-size:13px; font-family:'Segoe UI'; }}"
            f"QPushButton:hover {{ background:{BG_PANEL}; border-color:{STAGE_COLORS['comparison']}; }}"
            f"QPushButton:pressed {{ background:{STAGE_COLORS['comparison']}; color:{BG_DARK}; }}")
        btn_run.clicked.connect(self._do_update_runtime)
        layout.addWidget(btn_run)

        # Runtime bar chart
        layout.addWidget(self._section_header(
            "  RUNTIME ANALYSIS",
            "Elapsed time per method (ms) — green=fast  orange=medium  red=slow",
            STAGE_COLORS["comparison"]))

        self.analysis_panels["runtime_chart"] = AnalysisPanel(
            "Runtime Comparison", "All 9 methods benchmarked on current image",
            "comparison", figsize=(12.0, 4.5))
        self.analysis_panels["runtime_chart"].setMinimumHeight(360)
        layout.addWidget(self.analysis_panels["runtime_chart"])

        # Edge detection grid 
        layout.addWidget(self._section_header(
            "  EDGE DETECTION COMPARISON",
            "Prewitt · Sobel · Roberts · Ext. Sobel · Kirsch · Canny Scratch · Canny Library",
            STAGE_COLORS["comparison"]))

        self.analysis_panels["edge_compare_grid"] = AnalysisPanel(
            "Edge Detection Grid", "7 methods side-by-side",
            "comparison", figsize=(14.0, 7.0))
        self.analysis_panels["edge_compare_grid"].setMinimumHeight(480)
        layout.addWidget(self.analysis_panels["edge_compare_grid"])

        # Sharpening comparison 
        layout.addWidget(self._section_header(
            "  SHARPENING COMPARISON",
            "Original  ·  Laplacian Sharpened  ·  Unsharp Masking Result",
            STAGE_COLORS["comparison"]))

        self.analysis_panels["sharp_compare_grid"] = AnalysisPanel(
            "Sharpening Grid", "Original vs Laplacian vs USM",
            "comparison", figsize=(10.0, 4.0))
        self.analysis_panels["sharp_compare_grid"].setMinimumHeight(320)
        layout.addWidget(self.analysis_panels["sharp_compare_grid"])

        layout.addStretch()
        return scroll
    
    def _build_tab_harris(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()
        layout.addWidget(self._section_header("9. HARRIS CORNER DETECTION", "det(M) - α(trace(M))² with Gaussian smoothing", STAGE_COLORS["harris"]))
        
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setSpacing(14); cl.setContentsMargins(0,0,0,0)
        c_a, self.har_alpha_sld, self.har_alpha_lbl = self._slider_card("Harris α (×0.01)", 1, 25, 5, " → 0.05", STAGE_COLORS["harris"]); self.har_alpha_sld.valueChanged.connect(self._har_alpha_changed); cl.addWidget(c_a)
        c_s, self.har_sigma_sld, self.har_sigma_lbl = self._slider_card("Gaussian σ (×0.1)", 1, 30, 10, " → 1.0", STAGE_COLORS["harris"]); self.har_sigma_sld.valueChanged.connect(self._har_sigma_changed); cl.addWidget(c_s)
        c_t, self.har_thr_sld, self.har_thr_lbl = self._slider_card("Threshold (×0.01)", 1, 100, 5, " → 0.05", STAGE_COLORS["harris"]); self.har_thr_sld.valueChanged.connect(self._har_thr_changed); cl.addWidget(c_t)
        layout.addWidget(ctrl)
        
        self.panels["har_ix"] = ImagePanel("Gradient Ix", "Sobel X", "harris")
        self.panels["har_iy"] = ImagePanel("Gradient Iy", "Sobel Y", "harris")
        self.panels["har_a"] = ImagePanel("Tensor A", "Gaussian(Ix²)", "harris")
        self.panels["har_b"] = ImagePanel("Tensor B", "Gaussian(Iy²)", "harris")
        self.panels["har_c"] = ImagePanel("Tensor C", "Gaussian(Ix·Iy)", "harris")
        layout.addWidget(self._panel_row([self.panels["har_ix"], self.panels["har_iy"], self.panels["har_a"], self.panels["har_b"], self.panels["har_c"]]))
        
        self.panels["har_detm"] = ImagePanel("det(M)", "A·B - C²", "harris")
        self.panels["har_trace"] = ImagePanel("trace(M)", "A + B", "harris")
        self.panels["har_q"] = ImagePanel("Harris Q Map", "det - α·trace²", "harris")
        layout.addWidget(self._panel_row([self.panels["har_detm"], self.panels["har_trace"], self.panels["har_q"]]))
        
        self.panels["har_res_sc"] = ImagePanel("Result (Scratch)", "Manual math", "harris")
        self.panels["har_res_lib"] = ImagePanel("Result (Library)", "skimage", "harris")
        layout.addWidget(self._panel_row([self.panels["har_res_sc"], self.panels["har_res_lib"]]))
        
        layout.addWidget(self._section_header("  EVALUATION", "Corner Strength (Q-Value) Score Distribution", STAGE_COLORS["harris"]))
        self.analysis_panels["har_q_hist"] = AnalysisPanel("Q-Value Response", "Compares raw Corner Strength (Scratch) vs skimage response", "harris", figsize=(12.0, 3.5))
        self.analysis_panels["har_q_hist"].setMinimumHeight(280)
        layout.addWidget(self.analysis_panels["har_q_hist"])
        
        layout.addStretch(); return scroll

    def _build_tab_hough_line(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()
        layout.addWidget(self._section_header("10. HOUGH LINE TRANSFORM", "Hessian Normal Form: r = x·cos(θ) + y·sin(θ)", STAGE_COLORS["hough_line"]))
        
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setSpacing(14); cl.setContentsMargins(0,0,0,0)
        c_th, self.hl_theta_sld, self.hl_theta_lbl = self._slider_card("Theta Steps", 90, 360, 180, "", STAGE_COLORS["hough_line"]); self.hl_theta_sld.valueChanged.connect(self._hl_theta_changed); cl.addWidget(c_th)
        c_t, self.hl_thr_sld, self.hl_thr_lbl = self._slider_card("Threshold (Votes)", 10, 200, 50, "", STAGE_COLORS["hough_line"]); self.hl_thr_sld.valueChanged.connect(self._hl_thr_changed); cl.addWidget(c_t)
        layout.addWidget(ctrl)
        
        self.panels["hl_edge"] = ImagePanel("Canny Edge", "Input for Hough", "hough_line")
        self.panels["hl_acc_sc"] = ImagePanel("Accumulator (Scratch)", "Log scale heatmap", "hough_line")
        self.panels["hl_res_sc"] = ImagePanel("Detected Lines (Scratch)", "Manual drawing", "hough_line")
        layout.addWidget(self._panel_row([self.panels["hl_edge"], self.panels["hl_acc_sc"], self.panels["hl_res_sc"]]))
        
        self.panels["hl_acc_lib"] = ImagePanel("Accumulator (Library)", "skimage hough_line", "hough_line")
        self.panels["hl_res_lib"] = ImagePanel("Detected Lines (Library)", "skimage peaks", "hough_line")
        layout.addWidget(self._panel_row([self.panels["hl_acc_lib"], self.panels["hl_res_lib"]]))
        
        layout.addWidget(self._section_header("  EVALUATION", "Accumulator Vote Distribution Comparison", STAGE_COLORS["hough_line"]))
        self.analysis_panels["hl_acc_hist"] = AnalysisPanel("Accumulator Distribution", "Vote distribution (Scratch vs Lib)", "hough_line", figsize=(12.0, 3.5))
        self.analysis_panels["hl_acc_hist"].setMinimumHeight(280)
        layout.addWidget(self.analysis_panels["hl_acc_hist"])
        
        layout.addStretch(); return scroll

    def _build_tab_hough_circle(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()
        layout.addWidget(self._section_header("11. HOUGH CIRCLE TRANSFORM", "(x-a)² + (y-b)² = r²", STAGE_COLORS["hough_circle"]))
        
        ctrl = QWidget(); cl = QHBoxLayout(ctrl); cl.setSpacing(14); cl.setContentsMargins(0,0,0,0)
        c_r, self.hc_rad_sld, self.hc_rad_lbl = self._slider_card("Radius (px) [Scratch]", 5, 100, 15, "", STAGE_COLORS["hough_circle"]); self.hc_rad_sld.valueChanged.connect(self._hc_rad_changed); cl.addWidget(c_r)
        c_t, self.hc_thr_sld, self.hc_thr_lbl = self._slider_card("Threshold Fraction (×0.01)", 1, 100, 45, " → 0.45", STAGE_COLORS["hough_circle"]); self.hc_thr_sld.valueChanged.connect(self._hc_thr_changed); cl.addWidget(c_t)
        layout.addWidget(ctrl)
        
        self.panels["hc_edge"] = ImagePanel("Canny Edge", "Input for Hough", "hough_circle")
        self.panels["hc_acc_sc"] = ImagePanel("Accumulator (Scratch)", "Single radius heatmap", "hough_circle")
        self.panels["hc_res_sc"] = ImagePanel("Detected Circles (Scratch)", "Manual finding", "hough_circle")
        layout.addWidget(self._panel_row([self.panels["hc_edge"], self.panels["hc_acc_sc"], self.panels["hc_res_sc"]]))
        
        self.panels["hc_acc_lib"] = ImagePanel("Accumulator (Library)", "Multi-radii projection", "hough_circle")
        self.panels["hc_res_lib"] = ImagePanel("Detected Circles (Library)", "skimage peaks", "hough_circle")
        layout.addWidget(self._panel_row([self.panels["hc_acc_lib"], self.panels["hc_res_lib"]]))
        
        layout.addWidget(self._section_header("  EVALUATION", "Accumulator Vote Distribution Comparison", STAGE_COLORS["hough_circle"]))
        self.analysis_panels["hc_acc_hist"] = AnalysisPanel("Accumulator Distribution", "Vote distribution (Scratch vs Lib)", "hough_circle", figsize=(12.0, 3.5))
        self.analysis_panels["hc_acc_hist"].setMinimumHeight(280)
        layout.addWidget(self.analysis_panels["hc_acc_hist"])
        
        layout.addStretch(); return scroll

    def _build_tab_w11_comparison(self) -> QScrollArea:
        scroll, layout = self._scroll_tab()
        layout.addWidget(self._section_header("12. W11 RUNTIME and COMPARISON", "Benchmark Harris, Hough Line, Hough Circle", STAGE_COLORS["w11_compare"]))
        btn_run = QPushButton("▶  Run A3 Benchmarks Now"); btn_run.setMinimumHeight(44); btn_run.setStyleSheet(f"QPushButton {{ background:{BG_CARD}; color:{STAGE_COLORS['w11_compare']}; border:1px solid {STAGE_COLORS['w11_compare']}60; border-radius:8px; font-weight:bold; }} QPushButton:hover {{ background:{BG_PANEL}; border-color:{STAGE_COLORS['w11_compare']}; }}"); btn_run.clicked.connect(self._do_update_w11_runtime); layout.addWidget(btn_run)
        self.analysis_panels["w11_runtime_chart"] = AnalysisPanel("W11 Runtime Comparison", "Scratch vs Library timings", "w11_compare", figsize=(12.0, 4.5)); self.analysis_panels["w11_runtime_chart"].setMinimumHeight(360); layout.addWidget(self.analysis_panels["w11_runtime_chart"])
        self.analysis_panels["w11_compare_grid"] = AnalysisPanel("W11 Result Grid", "Side-by-side Visuals", "w11_compare", figsize=(14.0, 7.0)); self.analysis_panels["w11_compare_grid"].setMinimumHeight(480); layout.addWidget(self.analysis_panels["w11_compare_grid"])
        layout.addStretch(); return scroll

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Facial Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)")
        if not path:
            return
        try:
            if HAS_CV2:
                bgr = cv2.imread(path)
                if bgr is not None:
                    rgb = cv2.cvtColor(cv2.resize(bgr, (300, 300)), cv2.COLOR_BGR2RGB)
                    self.gray_img = manual_rgb2gray(rgb)
            else:
                from PIL import Image as PILImage
                pil = PILImage.open(path).convert("RGB").resize((300, 300))
                self.gray_img = manual_rgb2gray(np.array(pil))

            self.image_name = os.path.basename(path)
            # Reset caches
            self._last_canny_scratch  = None
            self._last_canny_lib      = None
            self._last_sharpening_lap = None
            self._last_sharpening_usm = None
            self._runtime_dirty       = True
            self._w11_runtime_dirty   = True

            self._update()
            
            self._canny_debounce.start(600)
            self._sharp_debounce.start(600)
            self._harris_debounce.start(600)
            self._hline_debounce.start(600)
            self._hcirc_debounce.start(600)
            
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

    def _canny_sigma_changed(self, val: int):
        self.canny_sigma = val / 10.0
        self.canny_sigma_lbl.setText(f"σ = {self.canny_sigma:.1f}")
        self._canny_debounce.start(350)
        self._hline_debounce.start(350)
        self._hcirc_debounce.start(350) 

    def _canny_tlo_changed(self, val: int):
        self.canny_t_lo = val / 100.0
        self.canny_tlo_lbl.setText(f"lo = {self.canny_t_lo:.2f}")
        self._canny_debounce.start(350)
        self._hline_debounce.start(350)   
        self._hcirc_debounce.start(350)   

    def _canny_thi_changed(self, val: int):
        self.canny_t_hi = val / 100.0
        self.canny_thi_lbl.setText(f"hi = {self.canny_t_hi:.2f}")
        self._canny_debounce.start(350)
        self._hline_debounce.start(350)   
        self._hcirc_debounce.start(350)

    def _lap_weight_changed(self, val: int):
        self.lap_weight = val / 10.0
        self.lap_weight_lbl.setText(f"w = {self.lap_weight:.1f}")
        self._sharp_debounce.start(350)

    def _lap_kernel_changed(self, text: str):
        if "H8"  in text: self.lap_kernel = "H8"
        elif "H12" in text: self.lap_kernel = "H12"
        else:              self.lap_kernel = "H4"
        self._sharp_debounce.start(350)

    def _usm_a_changed(self, val: int):
        self.usm_a = val / 10.0
        self.usm_a_lbl.setText(f"a = {self.usm_a:.1f}")
        self._sharp_debounce.start(350)

    def _usm_sigma_changed(self, val: int):
        self.usm_sigma = val / 10.0
        self.usm_sigma_lbl.setText(f"σ = {self.usm_sigma:.1f}")
        self._sharp_debounce.start(350)

    # Sliders A3
    def _har_alpha_changed(self, val: int): self.harris_alpha = val / 100.0; self.har_alpha_lbl.setText(f"Harris α = {self.harris_alpha:.2f}"); self._harris_debounce.start(350)
    def _har_sigma_changed(self, val: int): self.harris_sigma = val / 10.0; self.har_sigma_lbl.setText(f"σ = {self.harris_sigma:.1f}"); self._harris_debounce.start(350)
    def _har_thr_changed(self, val: int): self.harris_thr = val / 100.0; self.har_thr_lbl.setText(f"Thr = {self.harris_thr:.2f}"); self._harris_debounce.start(350)
    def _hl_theta_changed(self, val: int): self.hline_theta = val; self.hl_theta_lbl.setText(f"Theta Steps = {self.hline_theta}"); self._hline_debounce.start(350)
    def _hl_thr_changed(self, val: int): self.hline_thr = val; self.hl_thr_lbl.setText(f"Threshold = {self.hline_thr}"); self._hline_debounce.start(350)
    def _hc_rad_changed(self, val: int): self.hcirc_radius = val; self.hc_rad_lbl.setText(f"Radius = {self.hcirc_radius}"); self._hcirc_debounce.start(350)
    def _hc_thr_changed(self, val: int): self.hcirc_thr = val / 100.0; self.hc_thr_lbl.setText(f"Thr Frac = {self.hcirc_thr:.2f}"); self._hcirc_debounce.start(350)

    def _do_update_canny(self):
        """Compute Canny from scratch + library and refresh Tab 6 panels."""
        if self.gray_img is None:
            return
        try:
            gray = self.gray_img
            sc = canny_scratch(gray, self.canny_sigma, self.canny_t_lo, self.canny_t_hi)
            self._last_canny_scratch = sc

            self.panels["canny_smooth"].show_image(sc["smoothed"],       "gray")
            self.panels["canny_fx"].show_image(sc["fx"],                 "RdBu_r")
            self.panels["canny_fy"].show_image(sc["fy"],                 "PRGn_r")
            self.panels["canny_mag"].show_image(sc["magnitude"],         EDGE_CMAP, colorbar=True)
            self.panels["canny_angle"].show_image(sc["angle_disp"],      "hsv",     colorbar=True)
            self.panels["canny_quant"].show_image(sc["quantized_disp"],  CANNY_DIR_CMAP)
            self.analysis_panels["canny_color"].show_rgb_image(
                sc["color_rgb"],
                title="Direction Colors  R=Horiz  G=Diag↗  B=Vert  Y=Diag↘")
            self.panels["canny_nms"].show_image(sc["nms"],               EDGE_CMAP, colorbar=True)
            self.panels["canny_thresh"].show_image(sc["double_thresh_disp"], "gray")
            self.panels["canny_hys"].show_image(sc["hysteresis"],        "gray")

            # ── Library ─────────────────────────────────────────────────
            lib = canny_library(gray, self.canny_sigma, self.canny_t_lo, self.canny_t_hi)
            self._last_canny_lib = lib

            self.panels["canny_lib"].show_image(lib["result"], "gray")

            # Comparison histogram overlay
            self.analysis_panels["canny_compare"].show_comparison_hist_ogive(
                {"Scratch": sc["hysteresis"], "Library": lib["result"]},
                title=f"Canny Comparison  |  Scratch: {sc['elapsed']:.1f} ms  "
                      f"|  Library: {lib['elapsed']:.1f} ms  "
                      f"|  Scratch density: {sc['density']:.4f}  "
                      f"|  Library density: {lib['density']:.4f}")

            self._runtime_dirty = True
        except Exception as e:
            self.status.showMessage(f"[Canny error] {e}", 4000)

    def _do_update_sharpening(self):
        """Compute Laplacian + USM sharpening and refresh Tab 7 panels."""
        if self.gray_img is None:
            return
        try:
            gray = self.gray_img
            lap = laplacian_sharpening(gray, self.lap_weight, self.lap_kernel)
            self._last_sharpening_lap = lap

            self.panels["sharp_orig"].show_image(gray,               "gray")
            self.panels["sharp_lap_x"].show_image(lap["lap_x"],      EDGE_CMAP)
            self.panels["sharp_lap_y"].show_image(lap["lap_y"],      EDGE_CMAP)
            self.panels["sharp_lap_sum"].show_image(lap["lap_xy_sep"], EDGE_CMAP)
            self.panels["sharp_sep"].show_image(lap["sharp_sep"],    "gray")
            self.panels["sharp_lap_full"].show_image(lap["lap_full"], EDGE_CMAP)
            self.panels["sharp_full"].show_image(lap["sharp_full"],  "gray")

            # ── Unsharp Masking ──────────────────────────────────────────
            usm = unsharp_masking(gray, self.usm_a, self.usm_sigma)
            self._last_sharpening_usm = usm

            self.panels["sharp_usm_blur"].show_image(usm["blurred"],   "gray")
            self.panels["sharp_usm_mask"].show_image(usm["mask"],      "RdBu_r")
            self.panels["sharp_usm_result"].show_image(usm["sharpened"], "gray")

            # --- EVALUASI TAMBAHAN A2 ---
            prof_dict = {"Original": gray, "Laplacian": lap["sharp_full"], "USM": usm["sharpened"]}
            self.analysis_panels["sharp_profile"].show_intensity_profiles(prof_dict, row=gray.shape[0]//2, title="Edge Intensity Profile")

            self._runtime_dirty = True
        except Exception as e:
            self.status.showMessage(f"[Sharpening error] {e}", 4000)

    def _do_update_runtime(self):
        """Run ALL methods, measure runtimes, update Tab 8 panels."""
        if self.gray_img is None:
            return
        try:
            self.status.showMessage("⏳  Running all methods for comparison...", 0)
            QApplication.processEvents()

            sigma   = self.sigma_slider.value() / 10.0
            thr     = self.thr_slider.value() / 100.0
            enh_txt = self.enh_combo.currentText()
            if "CLAHE" in enh_txt:     enh_m = "CLAHE"
            elif "Histogram" in enh_txt: enh_m = "HE"
            elif "Contrast" in enh_txt:  enh_m = "CS"
            else:                        enh_m = "None"

            results = compute_all_methods(
                self.gray_img, sigma, thr,
                self.canny_sigma, self.canny_t_lo, self.canny_t_hi,
                self.lap_weight, self.lap_kernel,
                self.usm_a, self.usm_sigma,
                enh_m, self.enh_clahe_clip, self.cs_low, self.cs_high)

            # Runtime bar chart
            runtimes = {m: results[m]["elapsed"] for m in results}
            self._last_runtimes = runtimes
            self.analysis_panels["runtime_chart"].show_runtime_bars(runtimes)

            # Edge detection comparison grid (7 edge methods)
            edge_keys = ["Prewitt", "Sobel", "Roberts", "Extended Sobel",
                         "Kirsch", "Canny (Scratch)", "Canny (Library)"]
            edge_imgs = {k: results[k]["edge"] for k in edge_keys if k in results}
            self.analysis_panels["edge_compare_grid"].show_image_grid(
                edge_imgs,
                title="Edge Detection Comparison — All Methods",
                cmap="gray", cols=4)

            # Sharpening comparison grid
            sharp_imgs = {
                "Original (Gray)":     self.gray_img,
                "Laplacian Sharpened": results.get("Laplacian",        {}).get("edge"),
                "Unsharp Masking":     results.get("Unsharp Masking",  {}).get("edge"),
            }
            self.analysis_panels["sharp_compare_grid"].show_image_grid(
                sharp_imgs,
                title="Sharpening Comparison — Original vs Laplacian vs USM",
                cmap="gray", cols=3)

            self._runtime_dirty = False
            fastest = min(runtimes, key=runtimes.get)
            slowest = max(runtimes, key=runtimes.get)
            self.status.showMessage(
                f"✅  Runtime analysis done  |  "
                f"Fastest: {fastest} ({runtimes[fastest]:.1f} ms)  |  "
                f"Slowest: {slowest} ({runtimes[slowest]:.1f} ms)", 8000)
        except Exception as e:
            self.status.showMessage(f"[Runtime error] {e}", 5000)

    def _do_update_harris(self):
        if self.gray_img is None: return
        try:
            gray = self.gray_img
            sc = harris_corner_scratch(gray, self.harris_alpha, self.harris_sigma, self.harris_thr)
            self.panels["har_ix"].show_image(sc["Ix"], EDGE_CMAP); self.panels["har_iy"].show_image(sc["Iy"], EDGE_CMAP)
            self.panels["har_a"].show_image(sc["A"], EDGE_CMAP); self.panels["har_b"].show_image(sc["B"], EDGE_CMAP); self.panels["har_c"].show_image(sc["C"], EDGE_CMAP)
            self.panels["har_detm"].show_image(sc["detM"], "magma"); self.panels["har_trace"].show_image(sc["trace"], "magma"); self.panels["har_q"].show_image(sc["Q_map"], "magma")
            self.panels["har_res_sc"].show_rgb_image(sc["overlay"])
            lib = harris_corner_library(gray, self.harris_alpha, self.harris_sigma)
            self.panels["har_res_lib"].show_rgb_image(lib["overlay"])
            
            comp_dict = {"Scratch (Q-Map)": sc["Q_map"], "Library": lib["response"]}
            self.analysis_panels["har_q_hist"].show_comparison_hist_ogive(comp_dict, title="Harris Response Distribution")
            
            self._w11_runtime_dirty = True
        except Exception as e: self.status.showMessage(f"[Harris error] {e}", 4000)

    def _do_update_hline(self):
        if self.gray_img is None: return
        try:
            gray = self.gray_img
            sc = hough_line_scratch(gray, self.hline_theta, self.hline_thr, self.canny_sigma, self.canny_t_lo, self.canny_t_hi)
            self.panels["hl_edge"].show_image(sc["edges"], "gray"); self.panels["hl_acc_sc"].show_image(sc["accumulator"], "magma"); self.panels["hl_res_sc"].show_rgb_image(sc["result_rgb"])
            lib = hough_line_library(gray, self.canny_sigma, self.canny_t_lo, self.canny_t_hi)
            self.panels["hl_acc_lib"].show_image(lib["accumulator"], "magma"); self.panels["hl_res_lib"].show_rgb_image(lib["result_rgb"])
            
            # --- EVALUASI TAMBAHAN: Hough Line Accumulator Evaluation ---
            comp_dict = {"Scratch Accumulator": sc["accumulator"], "Library Accumulator": lib["accumulator"]}
            self.analysis_panels["hl_acc_hist"].show_comparison_hist_ogive(comp_dict, title="Hough Line Voting Evaluation")
            
            self._w11_runtime_dirty = True
        except Exception as e: self.status.showMessage(f"[Hough Line error] {e}", 4000)

    def _do_update_hcirc(self):
        if self.gray_img is None: return
        try:
            gray = self.gray_img
            sc = hough_circle_scratch(gray, self.hcirc_radius, self.hcirc_thr, self.canny_sigma, self.canny_t_lo, self.canny_t_hi)
            self.panels["hc_edge"].show_image(sc["edges"], "gray"); self.panels["hc_acc_sc"].show_image(sc["accumulator"], "magma"); self.panels["hc_res_sc"].show_rgb_image(sc["result_rgb"])
            lib = hough_circle_library(gray, self.canny_sigma, self.canny_t_lo, self.canny_t_hi)
            self.panels["hc_acc_lib"].show_image(lib["accumulator"], "magma"); self.panels["hc_res_lib"].show_rgb_image(lib["result_rgb"])
            
            comp_dict = {"Scratch Accumulator": sc["accumulator"], "Library Accumulator": lib["accumulator"]}
            self.analysis_panels["hc_acc_hist"].show_comparison_hist_ogive(comp_dict, title="Hough Circle Voting Evaluation")
            
            self._w11_runtime_dirty = True
        except Exception as e: self.status.showMessage(f"[Hough Circle error] {e}", 4000)

    def _do_update_w11_runtime(self):
        if self.gray_img is None: return
        try:
            self.status.showMessage("⏳  Running A3 methods...", 0); QApplication.processEvents()
            # Pass all parameters down
            results = compute_all_w11_methods(self.gray_img, self.harris_alpha, self.harris_sigma, self.harris_thr, self.hline_theta, self.hline_thr, self.hcirc_radius, self.hcirc_thr, self.canny_sigma, self.canny_t_lo, self.canny_t_hi)
            
            runtimes = {m: results[m]["elapsed"] for m in results}; self.analysis_panels["w11_runtime_chart"].show_runtime_bars(runtimes)
            grid_imgs = {
                "Harris (Scratch)": results["Harris (Scratch)"]["edge"], "Harris (Library)": results["Harris (Library)"]["edge"],
                "Hough Line (Scratch)": results["Hough Line (Scratch)"]["edge"], "Hough Line (Library)": results["Hough Line (Library)"]["edge"],
                "Hough Circle (Scratch)": results["Hough Circle (Scratch)"]["edge"], "Hough Circle (Library)": results["Hough Circle (Library)"]["edge"],
            }
            self.analysis_panels["w11_compare_grid"].show_image_grid(grid_imgs, title="Week 11 Visual Comparison", cols=3)
            self._w11_runtime_dirty = False
            self.status.showMessage(f"✅ A3 Benchmark done.", 8000)
        except Exception as e: self.status.showMessage(f"[A3 Runtime error] {e}", 5000)

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
            enh_method=enh_m,
            clahe_clip=self.enh_clahe_clip,
            cs_low=self.cs_low,
            cs_high=self.cs_high
        )
        self._last_result = res
        m_color = METHOD_COLORS.get(method, TEXT_COL)

        self.panels["acq_orig"].show_image(self.gray_img, "gray")
        self.panels["acq"].show_image(res["acq"], "gray")
        self.analysis_panels["acq_orig_ha"].show_hist_ogive(self.gray_img, label="Original Gray", color="#89B4FA")
        self.analysis_panels["acq_ha"].show_hist_ogive(res["acq"], label="Acquired Image", color="#FAB387")
        self.analysis_panels["acq_compare_ha"].show_comparison_hist_ogive(
            {"Original": self.gray_img, "Acquired": res["acq"]}, title="Original vs Acquired")

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

        self.status.showMessage(
            f"  {method}  |  Enh={enh_m}  |  σ={sigma:.2f}  |  thr={thr:.2f}"
            f"  |  density={res['density']:.4f}  |  ⏱ {res['elapsed']:.1f} ms")

        self._do_update_enhancement()
        self._autosave_timer.start(3000)

    def _do_update_enhancement(self):
        if self._last_result is None or self.gray_img is None:
            return

        acq = self._last_result["acq"]
        enhs = compute_all_enhancements(
            acq,
            clahe_clip=self.enh_clahe_clip,
            cs_low=self.cs_low,
            cs_high=self.cs_high,
        )

        key_map = {"CS": "enh_cs", "HE": "enh_he", "CLAHE": "enh_clahe"}
        for mname, panel_key in key_map.items():
            self.panels[panel_key].show_image(enhs[mname], "gray")

        ha_map = {
            "CS":    ("enh_cs_ha",    "Contrast Stretching"),
            "HE":    ("enh_he_ha",    "Histogram Equalization"),
            "CLAHE": ("enh_clahe_ha", "CLAHE (Adaptive)"),
        }
        for mname, (ap_key, lbl) in ha_map.items():
            self.analysis_panels[ap_key].show_hist_ogive(
                enhs[mname], label=lbl, color=ENH_COLORS[mname])

        metrics = {mname: compute_enhancement_metrics(acq, enhs[mname]) for mname in enhs}
        self.analysis_panels["enh_perf"].show_performance_eval(metrics)

    def _do_autosave(self):
        self._save_panels(auto=True)

    def _save_all_now(self):
        self._save_panels(auto=False)

    def _save_panels(self, auto: bool = False):
        if self._last_result is None:
            return
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
            if panel.save_to(os.path.join(subdir, fname)):
                ok += 1

        total = len(self.panels) + len(self.analysis_panels)
        label = "Auto-saved" if auto else "Saved"
        msg   = f"💾 {label} {ok}/{total} panels → {subdir}"
        self.autosave_lbl.setText(msg)
        self.autosave_lbl.setStyleSheet(f"color:{STAGE_COLORS['results']}; font-size:11px; padding-left:4px; font-family:'Segoe UI';")
        if not auto:
            self.status.showMessage(msg, 6000)