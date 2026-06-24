"""Dark theme: color palette, Matplotlib style, and Qt stylesheet"""

import matplotlib
matplotlib.use("QtAgg")

PAL = {
    "bg0":        "#050D1A",
    "bg1":        "#081526",
    "bg2":        "#0C1D38",
    "bg3":        "#0F2544",
    "bg4":        "#132D52",
    "sidebar":    "#06101E",
    "border":     "#1A3A60",
    "border2":    "#0E2540",
    "accent1":    "#1565C0",
    "accent2":    "#42A5F5",
    "accent3":    "#00BCD4",
    "accent4":    "#00E5FF",
    "green":      "#00E676",
    "orange":     "#FF9800",
    "red":        "#F44336",
    "text0":      "#EDF2FB",
    "text1":      "#B0C8E8",
    "text2":      "#6B8FBD",
    "text3":      "#3A608A",
    "tab_active": "#1565C0",
    "btn_save":   "#1A237E",
    "btn_sh":     "#283593",
}

MPL_STYLE = {
    "figure.facecolor": "#0C1D38",
    "axes.facecolor":   "#0F2544",
    "axes.edgecolor":   "#1A3A60",
    "axes.labelcolor":  "#B0C8E8",
    "axes.titlecolor":  "#EDF2FB",
    "text.color":       "#EDF2FB",
    "xtick.color":      "#B0C8E8",
    "ytick.color":      "#B0C8E8",
    "grid.color":       "#1A3A60",
    "grid.alpha":       0.4,
    "legend.facecolor": "#0F2544",
    "legend.edgecolor": "#1A3A60",
    "legend.labelcolor": "#EDF2FB",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
}
matplotlib.rcParams.update(MPL_STYLE)

DARK_QSS = """
/* ── Global reset ─────────────────────────────────────── */
QMainWindow { background-color: transparent; }
QWidget {
  color: %(text0)s;
  font-family: "Segoe UI","SF Pro Display","Helvetica Neue",Arial,sans-serif;
  font-size: 13px;
}
/* ── Scroll bars ──────────────────────────────────────── */
QScrollArea { border:none; background:transparent; }
QScrollBar:vertical { background:rgba(12,29,56,160); width:8px; border-radius:4px; }
QScrollBar::handle:vertical { background:%(border)s; min-height:20px; border-radius:4px; }
QScrollBar::handle:vertical:hover { background:%(accent1)s; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
QScrollBar:horizontal { background:rgba(12,29,56,160); height:8px; border-radius:4px; }
QScrollBar::handle:horizontal { background:%(border)s; min-width:20px; border-radius:4px; }
QScrollBar::handle:horizontal:hover { background:%(accent1)s; }
QScrollBar::add-line:horizontal,QScrollBar::sub-line:horizontal { width:0; }
/* ── Group boxes (global default) ────────────────────── */
QGroupBox {
  border: 1px solid %(border)s;
  border-radius: 8px;
  margin-top: 14px;
  padding-top: 10px;
  background: rgba(12,29,56,190);
}
QGroupBox::title {
  subcontrol-origin: margin;
  subcontrol-position: top left;
  left: 12px;
  padding: 0 6px;
  color: %(accent2)s;
  font-weight: 700;
  font-size: 12px;
}
/* ── Tab widget ───────────────────────────────────────── */
QTabWidget::pane {
  background: rgba(8,21,38,210);
  border: 1px solid %(border)s;
  border-radius: 0 6px 6px 6px;
}
QTabBar::tab {
  background: rgba(15,37,68,200);
  color: %(text2)s;
  border: 1px solid %(border2)s;
  border-bottom: none;
  padding: 7px 14px;
  border-radius: 4px 4px 0 0;
  margin-right: 2px;
  font-size: 12px;
  font-weight: 500;
}
QTabBar::tab:selected {
  background: %(tab_active)s;
  color: %(text0)s;
  border-color: %(accent1)s;
  font-weight: 700;
}
QTabBar::tab:hover:!selected { background: %(bg4)s; color: %(text1)s; }
QTabWidget QTabBar::tab { padding: 5px 10px; font-size: 11px; }
QTabWidget QTabBar::tab:selected { background: %(accent1)s; }
/* ── Inputs ───────────────────────────────────────────── */
QLineEdit, QTextEdit {
  background: rgba(19,45,82,230);
  color: %(text0)s;
  border: 1px solid %(border)s;
  border-radius: 4px;
  padding: 4px 8px;
  selection-background-color: %(accent1)s;
}
QLineEdit:focus, QTextEdit:focus { border-color: %(accent2)s; }
QSpinBox, QDoubleSpinBox {
  background: rgba(19,45,82,230);
  color: %(text0)s;
  border: 1px solid %(border)s;
  border-radius: 4px;
  padding: 3px 5px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border-color: %(accent2)s; }
QSpinBox::up-button,   QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
  width: 14px; background: %(bg3)s; border: none;
}
QComboBox {
  background: rgba(19,45,82,230);
  color: %(text0)s;
  border: 1px solid %(border)s;
  border-radius: 4px;
  padding: 4px 8px;
}
QComboBox:focus { border-color: %(accent2)s; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
  background: %(bg3)s;
  color: %(text0)s;
  border: 1px solid %(border)s;
  selection-background-color: %(accent1)s;
}
/* ── Checkbox ─────────────────────────────────────────── */
QCheckBox { color: %(text1)s; spacing: 6px; }
QCheckBox::indicator {
  width: 15px; height: 15px;
  border: 1px solid %(border)s;
  border-radius: 3px;
  background: rgba(19,45,82,200);
}
QCheckBox::indicator:checked  { background: %(accent1)s; border-color: %(accent2)s; }
QCheckBox::indicator:hover    { border-color: %(accent2)s; }
/* ── Buttons ──────────────────────────────────────────── */
QPushButton {
  background: rgba(15,37,68,220);
  color: %(text0)s;
  border: 1px solid %(border)s;
  border-radius: 5px;
  padding: 6px 14px;
  font-weight: 500;
}
QPushButton:hover   { background: %(bg4)s; border-color: %(accent2)s; }
QPushButton:pressed { background: %(accent1)s; }
QPushButton#btnRun {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #1B6F24, stop:1 #0F5217);
  color: #CCFFCC;
  border: 1px solid #2E7D32;
  font-weight: 700;
  font-size: 13px;
  padding: 10px 20px;
}
QPushButton#btnRun:hover {
  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
    stop:0 #2E7D32, stop:1 #1B5E20);
}
QPushButton#btnRun:disabled { background: %(bg3)s; color: %(text3)s; border-color: %(border2)s; }
QPushButton#btnSave {
  background: %(btn_save)s;
  color: %(accent2)s;
  border: 1px solid %(accent1)s;
  font-weight: 600;
}
QPushButton#btnSave:hover { background: %(btn_sh)s; }
QPushButton#btnBrowse {
  background: rgba(15,37,68,200);
  color: %(text1)s;
  border: 1px solid %(border)s;
  padding: 4px 10px;
  font-size: 11px;
}
QPushButton#btnBrowse:hover { border-color: %(accent2)s; color: %(text0)s; }
/* ── Progress / list ──────────────────────────────────── */
QProgressBar {
  background: rgba(15,37,68,180);
  border: 1px solid %(border)s;
  border-radius: 3px;
  height: 5px;
  color: transparent;
}
QProgressBar::chunk {
  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
    stop:0 %(accent1)s, stop:1 %(accent4)s);
  border-radius: 3px;
}
QListWidget {
  background: rgba(19,45,82,200);
  border: 1px solid %(border)s;
  border-radius: 4px;
}
QListWidget::item { padding:3px 6px; border-radius:3px; }
QListWidget::item:selected { background:%(accent1)s; }
QListWidget::item:hover { background:%(bg3)s; }
QStatusBar { background:%(bg0)s; color:%(text2)s;
  border-top:1px solid %(border2)s; font-size:11px; }
QSplitter::handle { background:%(border2)s; }
QSplitter::handle:hover { background:%(accent1)s; }
QToolTip { background:%(bg3)s; color:%(text0)s; border:1px solid %(accent2)s;
  padding:4px 8px; border-radius:4px; }
QLabel#sectionTitle { color:%(accent2)s; font-weight:700; font-size:12px;
  padding:2px 0; }
QLabel#imageTitle { color:%(accent4)s; font-weight:700; font-size:13px; }
""" % PAL
