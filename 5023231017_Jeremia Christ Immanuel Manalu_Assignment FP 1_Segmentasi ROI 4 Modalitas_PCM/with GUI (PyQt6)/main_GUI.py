import sys
import os
import warnings
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                             QComboBox, QTableView, QHeaderView, QScrollArea,
                             QFrame, QFileDialog, QMessageBox, QSpinBox, QGroupBox, QGridLayout,
                             QSizePolicy)
from PyQt6.QtCore import Qt, QAbstractTableModel

# Suppress scikit-image FutureWarnings in the terminal for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Matplotlib for PyQt6
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# Import All Custom Modules
from Fundus_retina_module import FundusProcessor
from Ultrasound_fetus_module import UltrasoundProcessor
from Xray_lung_pneumonia_module import XRayProcessor
from CTscan_chest_cancer_module_TIFF import view_all_tiffs, CTTiffProcessor
from CTscan_chest_cancer_module_DICOM import view_all_dicoms, CTProcessor as CTDicomProcessor

# 1. PANDAS TABLE MODEL (To display DataFrame in GUI)
class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._data.columns[col]
        return None

# 2. MAIN WINDOW GUI
class MedicalImagingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Image Processing and Radiomics Suite (FP 1)")
        self.setGeometry(50, 50, 1500, 900)
        self.apply_theme()

        # State Variables for caching module data
        self.fundus_state = {}
        self.usg_state = {}
        self.xray_state = {}
        self.ct_tiff_state = {}
        self.ct_dicom_state = {}

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        header = QLabel("MEDICAL IMAGE PROCESSING SUITE")
        header.setObjectName("HeaderLabel")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(header)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.build_fundus_tab()
        self.build_usg_tab()
        self.build_xray_tab()
        self.build_ctscan_tab()

    def apply_theme(self):
        # Navy (BG), Dark Yellow (Main), Magenta (Accent) Theme
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0A192F; color: #E6F1FF; font-family: 'Segoe UI', Arial, sans-serif; }
            QLabel#HeaderLabel { font-size: 24px; font-weight: bold; color: #FBC02D; padding: 10px; letter-spacing: 2px; }
            QTabWidget::pane { border: 1px solid #FBC02D; background: #112240; border-radius: 4px; }
            QTabBar::tab { background: #112240; color: #8892B0; padding: 10px 20px; font-weight: bold; border: 1px solid transparent; }
            QTabBar::tab:selected { background: #FBC02D; color: #0A192F; }
            QTabBar::tab:hover:!selected { background: #233554; color: #FBC02D; }
            QPushButton { background-color: #112240; border: 2px solid #FBC02D; color: #FBC02D; padding: 8px 15px; font-weight: bold; border-radius: 5px; }
            QPushButton:hover { background-color: #FBC02D; color: #0A192F; }
            QPushButton:disabled { border: 2px solid #8892B0; color: #8892B0; background-color: #0A192F; }
            QPushButton#AccentButton { background-color: #C2185B; border: 2px solid #C2185B; color: white; }
            QPushButton#AccentButton:hover { background-color: #E91E63; border: 2px solid #E91E63; }
            QFrame#ControlPanel { background-color: #112240; border-right: 2px solid #FBC02D; }
            QScrollArea { border: none; background-color: #0A192F; }
            QTableView { background-color: #233554; alternate-background-color: #112240; color: #E6F1FF; gridline-color: #FBC02D; border: 1px solid #FBC02D; }
            QHeaderView::section { background-color: #FBC02D; color: #0A192F; font-weight: bold; border: 1px solid #112240; padding: 4px; }
            QComboBox, QSpinBox { background-color: #233554; color: white; border: 1px solid #FBC02D; padding: 5px; }
            QGroupBox { border: 1px solid #FBC02D; margin-top: 15px; padding-top: 15px; font-weight: bold; color: #FBC02D; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)

    # Plot Rendering and Memory Management
    def clear_layout(self, layout):
        # Recursively clears widgets to prevent overlap and bugs
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
                elif item.layout() is not None:
                    self.clear_layout(item.layout())
                    item.layout().setParent(None)
                    item.layout().deleteLater()

    def render_figure(self, layout, fig):
        """
        Embeds a matplotlib figure into the PyQt layout safely.
        CRITICAL FIX: Enforces a minimum height dynamically based on figsize 
        to prevent aspect ratio distortion inside the ScrollArea.
        """
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 30) # Padding between plots
        
        try:
            fig.tight_layout()
        except Exception:
            pass

        canvas = FigureCanvas(fig)
        
        # Prevents grids from squishing
        fig_width, fig_height = fig.get_size_inches()
        dpi = fig.get_dpi()
        target_height = int(fig_height * dpi * 0.85) # Scale slightly to fit UI margins naturally
        
        canvas.setMinimumHeight(target_height)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        toolbar = NavigationToolbar(canvas, self)
        toolbar.setStyleSheet("background-color: #E6F1FF; color: black; border-radius: 3px;") 
        
        vbox.addWidget(toolbar)
        vbox.addWidget(canvas)
        layout.addWidget(container)
        
        canvas.draw()
        plt.close(fig) # Kills the figure instance to prevent RAM Memory Leaks

    # TAB 1: FUNDUS RETINA
    def build_fundus_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        control_panel = QFrame()
        control_panel.setObjectName("ControlPanel")
        control_panel.setFixedWidth(380)
        c_layout = QVBoxLayout(control_panel)
        
        btn_load = QPushButton("1. Load Fundus Dataset Dir")
        btn_load.clicked.connect(self.fundus_load_data)
        
        self.fundus_combo_img = QComboBox()
        btn_eval = QPushButton("2. Evaluate Enhancement")
        btn_eval.clicked.connect(self.fundus_evaluate)
        
        self.fundus_combo_enh = QComboBox()
        btn_seg = QPushButton("3. Segment and Extract Features")
        btn_seg.setObjectName("AccentButton")
        btn_seg.clicked.connect(self.fundus_segment)
        
        self.table_fundus = QTableView()
        
        c_layout.addWidget(btn_load)
        c_layout.addWidget(QLabel("Select Image to Process:"))
        c_layout.addWidget(self.fundus_combo_img)
        c_layout.addWidget(btn_eval)
        c_layout.addWidget(QLabel("Select Best Enhancement:"))
        c_layout.addWidget(self.fundus_combo_enh)
        c_layout.addWidget(btn_seg)
        c_layout.addWidget(QLabel("Extracted Features:"))
        c_layout.addWidget(self.table_fundus)
        
        scroll_fundus = QScrollArea()
        scroll_fundus.setWidgetResizable(True)
        content_fundus = QWidget()
        self.vbox_fundus = QVBoxLayout(content_fundus)
        scroll_fundus.setWidget(content_fundus)
        
        layout.addWidget(control_panel)
        layout.addWidget(scroll_fundus)
        self.tabs.addTab(tab, "👁️ Fundus Retina")

    def fundus_load_data(self):
        try:
            path_norm, _ = QFileDialog.getOpenFileName(self, "Select Normal Fundus JPG", "", "Images (*.jpg *.jpeg *.png)")
            path_reti, _ = QFileDialog.getOpenFileName(self, "Select Retinography JPG", "", "Images (*.jpg *.jpeg *.png)")
            export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            
            if not path_norm or not path_reti or not export_dir: return

            crop_coords = {"Normal Fundus": (1800, 2200, 870, 1250), "Retinography": (950, 1370, 620, 900)}
            
            self.fundus_processor = FundusProcessor(path_norm, path_reti, crop_coords, export_dir)
            self.fundus_combo_img.clear()
            self.fundus_combo_img.addItems(self.fundus_processor.dataset.keys())
            QMessageBox.information(self, "Success", "Fundus dataset loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{str(e)}")

    def fundus_evaluate(self):
        if not hasattr(self, 'fundus_processor'): return
        try:
            plt.close('all') # Clear memory buffer
            self.clear_layout(self.vbox_fundus)
            title = self.fundus_combo_img.currentText()
            img_rgb_full = self.fundus_processor.dataset[title]
            
            img_red, fig1, fig_hist = self.fundus_processor.step1_plot_and_convert_channels(img_rgb_full, title)
            crop_red, crop_rgb, coords = self.fundus_processor.step2_hardcoded_crop(img_red, img_rgb_full, title)
            enhanced_dict, df_metrics, fig2, fig_bar = self.fundus_processor.step3_evaluate_enhancement(crop_red, title)
            
            self.fundus_state = {'crop_red': crop_red, 'crop_rgb': crop_rgb, 'coords': coords, 
                                 'img_rgb_full': img_rgb_full, 'enhanced_dict': enhanced_dict, 'title': title}
            
            self.fundus_combo_enh.clear()
            self.fundus_combo_enh.addItems(enhanced_dict.keys())
            
            self.render_figure(self.vbox_fundus, fig1)
            self.render_figure(self.vbox_fundus, fig_hist)
            self.render_figure(self.vbox_fundus, fig2)
            self.render_figure(self.vbox_fundus, fig_bar)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def fundus_segment(self):
        if not self.fundus_state: return
        try:
            plt.close('all') # Clear memory buffer
            method_name = self.fundus_combo_enh.currentText()
            best_img = self.fundus_state['enhanced_dict'][method_name]
            
            df_features, fig_mask, fig_over = self.fundus_processor.step4_segment_and_extract(
                best_img, self.fundus_state['crop_red'], self.fundus_state['crop_rgb'], 
                self.fundus_state['img_rgb_full'], self.fundus_state['coords'], self.fundus_state['title'], method_name)
            
            self.render_figure(self.vbox_fundus, fig_mask)
            self.render_figure(self.vbox_fundus, fig_over)
            self.table_fundus.setModel(PandasModel(df_features))
            QMessageBox.information(self, "Exported", "Features extracted and saved to Excel!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


    # TAB 2: ULTRASOUND FETUS
    def build_usg_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        control_panel = QFrame(); control_panel.setObjectName("ControlPanel"); control_panel.setFixedWidth(380)
        c_layout = QVBoxLayout(control_panel)
        
        btn_load = QPushButton("1. Load Ultrasound Dataset")
        btn_load.clicked.connect(self.usg_load_data)
        self.usg_combo_img = QComboBox()
        btn_eval = QPushButton("2. Filter and Evaluate")
        btn_eval.clicked.connect(self.usg_evaluate)
        self.usg_combo_enh = QComboBox()
        btn_seg = QPushButton("3. Multi-Otsu and Extract (Ellipse)")
        btn_seg.setObjectName("AccentButton")
        btn_seg.clicked.connect(self.usg_segment)
        self.table_usg = QTableView()
        
        c_layout.addWidget(btn_load)
        c_layout.addWidget(QLabel("Select Image to Process:"))
        c_layout.addWidget(self.usg_combo_img)
        c_layout.addWidget(btn_eval)
        c_layout.addWidget(QLabel("Select Best Enhancement:"))
        c_layout.addWidget(self.usg_combo_enh)
        c_layout.addWidget(btn_seg)
        c_layout.addWidget(QLabel("Fetal Biometry Features:"))
        c_layout.addWidget(self.table_usg)
        
        scroll_usg = QScrollArea(); scroll_usg.setWidgetResizable(True)
        content_usg = QWidget()
        self.vbox_usg = QVBoxLayout(content_usg)
        scroll_usg.setWidget(content_usg)
        
        layout.addWidget(control_panel)
        layout.addWidget(scroll_usg)
        self.tabs.addTab(tab, "👶 Ultrasound Fetus")

    def usg_load_data(self):
        try:
            path1, _ = QFileDialog.getOpenFileName(self, "Select USG Image 1", "", "Images (*.png *.jpg)")
            path2, _ = QFileDialog.getOpenFileName(self, "Select USG Image 2", "", "Images (*.png *.jpg)")
            export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if not path1 or not path2 or not export_dir: return

            crop_coords = {"Fetus 166_HC": (155, 730, 55, 478), "Fetus 87_HC": (101, 556, 109, 456)}
            self.usg_processor = UltrasoundProcessor(path1, path2, crop_coords, export_dir)
            self.usg_combo_img.clear()
            self.usg_combo_img.addItems(self.usg_processor.dataset.keys())
            QMessageBox.information(self, "Success", "USG dataset loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def usg_evaluate(self):
        if not hasattr(self, 'usg_processor'): return
        try:
            plt.close('all') # Clear memory buffer
            self.clear_layout(self.vbox_usg)
            title = self.usg_combo_img.currentText()
            img_gray_full = self.usg_processor.dataset[title]
            
            fig1 = self.usg_processor.step1_plot_initial(img_gray_full, title)
            img_crop, img_restored, coords = self.usg_processor.step2_crop_and_restore(img_gray_full, title)
            enhanced_dict, df_metrics, fig2, fig_bar = self.usg_processor.step3_evaluate_enhancement(img_restored, title)
            
            self.usg_state = {'img_crop': img_crop, 'img_gray_full': img_gray_full, 'coords': coords, 
                              'enhanced_dict': enhanced_dict, 'title': title}
            
            self.usg_combo_enh.clear()
            self.usg_combo_enh.addItems(enhanced_dict.keys())
            
            self.render_figure(self.vbox_usg, fig1)
            self.render_figure(self.vbox_usg, fig2)
            self.render_figure(self.vbox_usg, fig_bar)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def usg_segment(self):
        if not self.usg_state: return
        try:
            plt.close('all') # Clear memory buffer
            method_name = self.usg_combo_enh.currentText()
            best_img = self.usg_state['enhanced_dict'][method_name]
            
            df_features, fig_otsu, fig_class_over, fig_final = self.usg_processor.step4_segment_and_extract(
                best_img, self.usg_state['img_crop'], self.usg_state['img_gray_full'], 
                self.usg_state['coords'], self.usg_state['title'], method_name)
            
            self.render_figure(self.vbox_usg, fig_otsu)
            self.render_figure(self.vbox_usg, fig_class_over)
            self.render_figure(self.vbox_usg, fig_final)
            self.table_usg.setModel(PandasModel(df_features))
            QMessageBox.information(self, "Exported", "Features extracted and saved to Excel!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


    # TAB 3: CHEST X-RAY
    def build_xray_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        control_panel = QFrame(); control_panel.setObjectName("ControlPanel"); control_panel.setFixedWidth(380)
        c_layout = QVBoxLayout(control_panel)
        
        btn_load = QPushButton("1. Load X-Ray Images and Dir")
        btn_load.clicked.connect(self.xray_load_data)
        self.xray_combo_img = QComboBox()
        btn_eval = QPushButton("2. Crop and Evaluate")
        btn_eval.clicked.connect(self.xray_evaluate)
        self.xray_combo_enh = QComboBox()
        btn_seg = QPushButton("3. Organic Edge Segment and Extract")
        btn_seg.setObjectName("AccentButton")
        btn_seg.clicked.connect(self.xray_segment)
        self.table_xray = QTableView()
        
        c_layout.addWidget(btn_load)
        c_layout.addWidget(QLabel("Select Image to Process:"))
        c_layout.addWidget(self.xray_combo_img)
        c_layout.addWidget(btn_eval)
        c_layout.addWidget(QLabel("Select Best Enhancement:"))
        c_layout.addWidget(self.xray_combo_enh)
        c_layout.addWidget(btn_seg)
        c_layout.addWidget(QLabel("Cardiac Silhouette Features:"))
        c_layout.addWidget(self.table_xray)
        
        scroll_xray = QScrollArea(); scroll_xray.setWidgetResizable(True)
        content_xray = QWidget()
        self.vbox_xray = QVBoxLayout(content_xray)
        scroll_xray.setWidget(content_xray)
        
        layout.addWidget(control_panel)
        layout.addWidget(scroll_xray)
        self.tabs.addTab(tab, "🩻 Chest X-Ray")

    def xray_load_data(self):
        try:
            paths, _ = QFileDialog.getOpenFileNames(self, "Select X-Ray Images", "", "Images (*.jpeg *.jpg *.png)")
            export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if len(paths) == 0 or not export_dir: return

            image_paths_dict = {os.path.basename(p): p for p in paths}
            
            # Mapping Specific X-Ray Coordinates
            crop_coordinates = {}
            for filename in image_paths_dict.keys():
                fn_lower = filename.lower()
                if "1440" in fn_lower:
                    crop_coordinates[filename] = (540, 1249, 302, 1053)
                elif "1442" in fn_lower:
                    crop_coordinates[filename] = (600, 1175, 353, 1030)
                elif "virus" in fn_lower:
                    crop_coordinates[filename] = (579, 1295, 306, 1075)
                elif "bacteria" in fn_lower:
                    crop_coordinates[filename] = (424, 935, 86, 602)
                else:
                    # Fallback default if filename is entirely different
                    crop_coordinates[filename] = (540, 1249, 302, 1053) 
            
            self.xray_processor = XRayProcessor(image_paths_dict, crop_coordinates, export_dir)
            self.xray_combo_img.clear()
            self.xray_combo_img.addItems(self.xray_processor.dataset.keys())
            QMessageBox.information(self, "Success", "X-Ray dataset loaded with precise crop coordinates!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def xray_evaluate(self):
        if not hasattr(self, 'xray_processor'): return
        try:
            plt.close('all') # Clear memory buffer
            self.clear_layout(self.vbox_xray)
            title = self.xray_combo_img.currentText()
            img_gray_full = self.xray_processor.dataset[title]
            
            fig1 = self.xray_processor.step1_plot_initial(img_gray_full, title)
            
            # Capture thoracic_width and fig_lung from step2_crop
            img_crop, coords, thoracic_width, fig_lung = self.xray_processor.step2_crop(img_gray_full, title)
            
            enhanced_dict, df_metrics, fig2, fig_bar = self.xray_processor.step3_evaluate_enhancement(img_crop, title)
            
            # Store thoracic_width in the state
            self.xray_state = {'img_crop': img_crop, 'img_gray_full': img_gray_full, 'coords': coords, 
                               'thoracic_width': thoracic_width,
                               'enhanced_dict': enhanced_dict, 'title': title}
            
            self.xray_combo_enh.clear()
            self.xray_combo_enh.addItems(enhanced_dict.keys())
            
            self.render_figure(self.vbox_xray, fig1)
            self.render_figure(self.vbox_xray, fig_lung) 
            self.render_figure(self.vbox_xray, fig2)
            self.render_figure(self.vbox_xray, fig_bar)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def xray_segment(self):
        if not self.xray_state: return
        try:
            plt.close('all') # Clear memory buffer
            method_name = self.xray_combo_enh.currentText()
            best_img = self.xray_state['enhanced_dict'][method_name]
            
            # Pass thoracic_width to step4_segment_and_extract
            df_features, fig_otsu, fig_prog, fig_final = self.xray_processor.step4_segment_and_extract(
                best_img, self.xray_state['img_crop'], self.xray_state['img_gray_full'], 
                self.xray_state['coords'], self.xray_state['thoracic_width'],
                self.xray_state['title'], method_name)
            
            self.render_figure(self.vbox_xray, fig_otsu)
            self.render_figure(self.vbox_xray, fig_prog)
            self.render_figure(self.vbox_xray, fig_final)
            self.table_xray.setModel(PandasModel(df_features))
            QMessageBox.information(self, "Exported", "Features extracted and saved to Excel!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


    # TAB 4: CT-SCAN (TIFF and DICOM)
    def build_ctscan_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        ct_tabs = QTabWidget()
        
        # SUB-TAB A: TIFF
        tab_tiff = QWidget()
        layout_tiff = QHBoxLayout(tab_tiff)
        panel_tiff = QFrame(); panel_tiff.setObjectName("ControlPanel"); panel_tiff.setFixedWidth(380)
        c_tiff = QVBoxLayout(panel_tiff)
        
        btn_gallery_tiff = QPushButton("1. Open TIFF Gallery Viewer")
        btn_gallery_tiff.setObjectName("AccentButton")
        btn_gallery_tiff.clicked.connect(self.ct_tiff_gallery)
        
        grp_idx_tiff = QGroupBox("Select Slice Indexes")
        lyt_idx_tiff = QGridLayout()
        self.spin_tiff_1 = QSpinBox(); self.spin_tiff_1.setMaximum(9999)
        self.spin_tiff_2 = QSpinBox(); self.spin_tiff_2.setMaximum(9999)
        lyt_idx_tiff.addWidget(QLabel("Index 1:"), 0, 0); lyt_idx_tiff.addWidget(self.spin_tiff_1, 0, 1)
        lyt_idx_tiff.addWidget(QLabel("Index 2:"), 1, 0); lyt_idx_tiff.addWidget(self.spin_tiff_2, 1, 1)
        grp_idx_tiff.setLayout(lyt_idx_tiff)

        btn_load_tiff = QPushButton("2. Load Selected Indexes")
        btn_load_tiff.clicked.connect(self.ct_tiff_load)
        self.ct_tiff_combo_img = QComboBox()
        btn_eval_tiff = QPushButton("3. Evaluate Enhancement")
        btn_eval_tiff.clicked.connect(self.ct_tiff_evaluate)
        self.ct_tiff_combo_enh = QComboBox()
        btn_seg_tiff = QPushButton("4. Extract TIFF Features")
        btn_seg_tiff.setObjectName("AccentButton")
        btn_seg_tiff.clicked.connect(self.ct_tiff_segment)
        self.table_ct_tiff = QTableView()
        
        c_tiff.addWidget(btn_gallery_tiff)
        c_tiff.addWidget(grp_idx_tiff)
        c_tiff.addWidget(btn_load_tiff)
        c_tiff.addWidget(QLabel("Select Loaded Slice:"))
        c_tiff.addWidget(self.ct_tiff_combo_img)
        c_tiff.addWidget(btn_eval_tiff)
        c_tiff.addWidget(QLabel("Select Enhancement:"))
        c_tiff.addWidget(self.ct_tiff_combo_enh)
        c_tiff.addWidget(btn_seg_tiff)
        c_tiff.addWidget(self.table_ct_tiff)
        
        scroll_tiff = QScrollArea(); scroll_tiff.setWidgetResizable(True)
        content_tiff = QWidget(); self.vbox_ct_tiff = QVBoxLayout(content_tiff)
        scroll_tiff.setWidget(content_tiff)
        layout_tiff.addWidget(panel_tiff); layout_tiff.addWidget(scroll_tiff)
        
        # SUB-TAB B: DICOM
        tab_dicom = QWidget()
        layout_dicom = QHBoxLayout(tab_dicom)
        panel_dicom = QFrame(); panel_dicom.setObjectName("ControlPanel"); panel_dicom.setFixedWidth(380)
        c_dicom = QVBoxLayout(panel_dicom)
        
        btn_gallery_dicom = QPushButton("1. Open DICOM Interactive Viewer")
        btn_gallery_dicom.setObjectName("AccentButton")
        btn_gallery_dicom.clicked.connect(self.ct_dicom_gallery)
        
        grp_idx_dicom = QGroupBox("Select Slice Indexes")
        lyt_idx_dicom = QGridLayout()
        self.spin_dicom_1 = QSpinBox(); self.spin_dicom_1.setMaximum(9999)
        self.spin_dicom_2 = QSpinBox(); self.spin_dicom_2.setMaximum(9999)
        lyt_idx_dicom.addWidget(QLabel("Index 1:"), 0, 0); lyt_idx_dicom.addWidget(self.spin_dicom_1, 0, 1)
        lyt_idx_dicom.addWidget(QLabel("Index 2:"), 1, 0); lyt_idx_dicom.addWidget(self.spin_dicom_2, 1, 1)
        grp_idx_dicom.setLayout(lyt_idx_dicom)

        btn_load_dicom = QPushButton("2. Load Selected Indexes")
        btn_load_dicom.clicked.connect(self.ct_dicom_load)
        self.ct_dicom_combo_img = QComboBox()
        btn_seg_dicom = QPushButton("3. Process Windowing and Segment")
        btn_seg_dicom.setObjectName("AccentButton")
        btn_seg_dicom.clicked.connect(self.ct_dicom_process)
        self.table_ct_dicom = QTableView()
        
        c_dicom.addWidget(btn_gallery_dicom)
        c_dicom.addWidget(grp_idx_dicom)
        c_dicom.addWidget(btn_load_dicom)
        c_dicom.addWidget(QLabel("Select Loaded Slice:"))
        c_dicom.addWidget(self.ct_dicom_combo_img)
        c_dicom.addWidget(btn_seg_dicom)
        c_dicom.addWidget(self.table_ct_dicom)
        
        scroll_dicom = QScrollArea(); scroll_dicom.setWidgetResizable(True)
        content_dicom = QWidget(); self.vbox_ct_dicom = QVBoxLayout(content_dicom)
        scroll_dicom.setWidget(content_dicom)
        layout_dicom.addWidget(panel_dicom); layout_dicom.addWidget(scroll_dicom)
        
        ct_tabs.addTab(tab_tiff, "TIFF Format")
        ct_tabs.addTab(tab_dicom, "DICOM Format")
        layout.addWidget(ct_tabs)
        self.tabs.addTab(tab, "🖥️ CT-Scan Cardiac")

    # CT-TIFF LOGIC
    def ct_tiff_gallery(self):
        self.tiff_dir = QFileDialog.getExistingDirectory(self, "Select TIFF Dataset Directory")
        if not self.tiff_dir: return
        self.valid_tiffs = view_all_tiffs(self.tiff_dir)
        QMessageBox.information(self, "Gallery Closed", "Please enter the index numbers you noted into the boxes.")

    def ct_tiff_load(self):
        if not hasattr(self, 'valid_tiffs'): return
        try:
            export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if not export_dir: return
            idx1, idx2 = self.spin_tiff_1.value(), self.spin_tiff_2.value()
            selected_files = [self.valid_tiffs[idx1], self.valid_tiffs[idx2]]
            
            # Mapping Specific TIFF Coordinates based on selection sequence
            crop_coords = {}
            if len(selected_files) > 0:
                crop_coords[selected_files[0]] = (226, 330, 170, 310)
            if len(selected_files) > 1:
                crop_coords[selected_files[1]] = (177, 380, 129, 300) 
                
            self.ct_tiff_processor = CTTiffProcessor(self.tiff_dir, selected_files, crop_coords, export_dir)
            self.ct_tiff_combo_img.clear()
            self.ct_tiff_combo_img.addItems(selected_files)
            QMessageBox.information(self, "Success", "TIFF dataset loaded with precise crop coordinates!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def ct_tiff_evaluate(self):
        if not hasattr(self, 'ct_tiff_processor'): return
        try:
            plt.close('all') # Clear memory buffer
            self.clear_layout(self.vbox_ct_tiff)
            title = self.ct_tiff_combo_img.currentText()
            img_orig_full = self.ct_tiff_processor.dataset[title]
            
            img_crop, coords, fig1 = self.ct_tiff_processor.step1_plot_initial_and_crop(img_orig_full, title)
            enhanced_dict, df_metrics, fig2, fig_bar = self.ct_tiff_processor.step2_evaluate_enhancement(img_crop, title)
            
            self.ct_tiff_state = {'img_crop': img_crop, 'img_orig_full': img_orig_full, 'coords': coords, 
                                  'enhanced_dict': enhanced_dict, 'title': title}
            
            self.ct_tiff_combo_enh.clear()
            self.ct_tiff_combo_enh.addItems(enhanced_dict.keys())
            
            self.render_figure(self.vbox_ct_tiff, fig1)
            self.render_figure(self.vbox_ct_tiff, fig2)
            self.render_figure(self.vbox_ct_tiff, fig_bar)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def ct_tiff_segment(self):
        if not self.ct_tiff_state: return
        try:
            plt.close('all') # Clear memory buffer
            method_name = self.ct_tiff_combo_enh.currentText()
            best_img = self.ct_tiff_state['enhanced_dict'][method_name]
            
            df_feat, fig_otsu, fig_prog, fig_final = self.ct_tiff_processor.step3_segment_and_extract(
                best_img, self.ct_tiff_state['img_crop'], self.ct_tiff_state['img_orig_full'], 
                self.ct_tiff_state['coords'], self.ct_tiff_state['title'], method_name)
            
            self.render_figure(self.vbox_ct_tiff, fig_otsu)
            self.render_figure(self.vbox_ct_tiff, fig_prog)
            self.render_figure(self.vbox_ct_tiff, fig_final)
            self.table_ct_tiff.setModel(PandasModel(df_feat))
            QMessageBox.information(self, "Exported", "TIFF Features saved to Excel!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # CT-DICOM LOGIC
    def ct_dicom_gallery(self):
        self.dicom_dir = QFileDialog.getExistingDirectory(self, "Select DICOM Dataset Directory")
        if not self.dicom_dir: return
        self.valid_dicoms = view_all_dicoms(self.dicom_dir)
        QMessageBox.information(self, "Gallery Closed", "Please enter the index numbers you noted into the boxes.")

    def ct_dicom_load(self):
        if not hasattr(self, 'valid_dicoms'): return
        try:
            export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if not export_dir: return
            idx1, idx2 = self.spin_dicom_1.value(), self.spin_dicom_2.value()
            selected_files = [self.valid_dicoms[idx1], self.valid_dicoms[idx2]]
            
            self.ct_dicom_processor = CTDicomProcessor(self.dicom_dir, selected_files, export_dir)
            self.ct_dicom_combo_img.clear()
            self.ct_dicom_combo_img.addItems(selected_files)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def ct_dicom_process(self):
        if not hasattr(self, 'ct_dicom_processor'): return
        try:
            plt.close('all') # Clear memory buffer
            self.clear_layout(self.vbox_ct_dicom)
            title = self.ct_dicom_combo_img.currentText()
            hu_img = self.ct_dicom_processor.dataset[title]
            meta = self.ct_dicom_processor.metadata[title]
            
            df_feat, fig_otsu, fig_prog, fig_final = self.ct_dicom_processor.step_process_image(hu_img, meta, title)
            
            self.render_figure(self.vbox_ct_dicom, fig_otsu)
            self.render_figure(self.vbox_ct_dicom, fig_prog)
            self.render_figure(self.vbox_ct_dicom, fig_final)
            self.table_ct_dicom.setModel(PandasModel(df_feat))
            QMessageBox.information(self, "Exported", "DICOM Features saved to Excel!")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    plt.ioff() 
    window = MedicalImagingGUI()
    window.show()
    sys.exit(app.exec())