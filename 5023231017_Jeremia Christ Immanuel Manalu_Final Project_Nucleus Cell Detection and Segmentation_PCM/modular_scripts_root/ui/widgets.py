"""Reusable custom PyQt6 widgets: canvas, background, sidebar section, log, badge"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from PyQt6.QtWidgets import (
    QWidget, QLabel, QScrollArea, QVBoxLayout, QHBoxLayout,
    QTextEdit, QSizePolicy, QGroupBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap, QPainter, QBrush, QLinearGradient

from .theme import PAL


class MatplotlibCanvas(QWidget):
    """Matplotlib figure + NavigationToolbar inside a scroll area."""

    def __init__(self, parent=None, figsize=(10, 6)):
        super().__init__(parent)
        self._fig    = Figure(figsize=figsize, facecolor=PAL["bg2"])
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)

        self._toolbar.setFixedHeight(32)
        self._toolbar.setStyleSheet(
            f"background:{PAL['bg2']}; border:none; border-bottom:1px solid {PAL['border2']};")

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._canvas)
        self._scroll.setStyleSheet("background:transparent; border:none;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._scroll)

    def update_figure(self, fig: Figure):
        """Replace the current figure, keeps QScrollArea alive to avoid the deleted-C++-object crash."""
        plt.close(self._fig)
        self._fig = fig

        self._canvas = FigureCanvas(fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)

        new_toolbar = NavigationToolbar(self._canvas, self)
        new_toolbar.setFixedHeight(32)
        new_toolbar.setStyleSheet(
            f"background:{PAL['bg2']}; border:none;"
            f"border-bottom:1px solid {PAL['border2']};")

        layout = self.layout()
        old_item = layout.takeAt(0)
        if old_item and old_item.widget():
            old_item.widget().deleteLater()
        layout.insertWidget(0, new_toolbar)
        self._toolbar = new_toolbar

        self._scroll.setWidget(self._canvas)

        w = int(fig.get_figwidth()  * fig.dpi)
        h = int(fig.get_figheight() * fig.dpi)
        self._canvas.setMinimumSize(w, h)
        self._canvas.draw()

    def show_placeholder(self, text="Run processing to view results"):
        self._fig.clf()
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(PAL["bg3"])
        ax.text(0.5, 0.5, text, ha="center", va="center",
                fontsize=13, color=PAL["text2"], transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(PAL["border2"])
        self._canvas.draw()


class BgCentralWidget(QWidget):
    """
    Central widget whose paintEvent draws the background image or a dark gradient.
    Child containers with WA_TranslucentBackground + rgba() QSS backgrounds
    produce a glass / frosted-glass effect where the background shows through.
    """
    _OVERLAY_ALPHA = 155

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bg_pixmap = None
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

    def set_bg_pixmap(self, px):
        self._bg_pixmap = px
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        if self._bg_pixmap and not self._bg_pixmap.isNull():
            scaled = self._bg_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (scaled.width()  - w) // 2
            y = (scaled.height() - h) // 2
            p.drawPixmap(0, 0, scaled, x, y, w, h)
            p.fillRect(0, 0, w, h, QColor(5, 13, 26, self._OVERLAY_ALPHA))
        else:
            grad = QLinearGradient(0, 0, w, h)
            grad.setColorAt(0.0, QColor(5,  13, 26))
            grad.setColorAt(0.5, QColor(8,  21, 38))
            grad.setColorAt(1.0, QColor(12, 29, 56))
            p.fillRect(0, 0, w, h, QBrush(grad))
        p.end()


class SidebarSection(QGroupBox):
    """Parameter section with an accent-coloured left border and semi-transparent background."""

    _SECTION_QSS = (
        "QGroupBox {"
        "  border: 1px solid #2A5C9A;"
        "  border-left: 3px solid " + PAL["accent2"] + ";"
        "  border-radius: 6px;"
        "  margin-top: 14px;"
        "  padding-top: 10px;"
        "  background: rgba(8, 21, 50, 190);"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  subcontrol-position: top left;"
        "  left: 14px;"
        "  padding: 0 6px;"
        "  color: " + PAL["accent2"] + ";"
        "  font-weight: 700;"
        "  font-size: 12px;"
        "}"
    )

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(self._SECTION_QSS)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._body = QWidget()
        self._body.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._layout = QVBoxLayout(self._body)
        self._layout.setContentsMargins(8, 4, 8, 8)
        self._layout.setSpacing(5)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._body)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def body_layout(self):
        return self._layout

    def add_row(self, label: str, widget: QWidget):
        row = QWidget()
        row.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        hl  = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)
        lbl = QLabel(label)
        lbl.setFixedWidth(120)
        lbl.setStyleSheet(f"color:{PAL['text1']}; font-size:12px;")
        hl.addWidget(lbl)
        hl.addWidget(widget)
        self._layout.addWidget(row)

    def add_widget(self, widget: QWidget):
        self._layout.addWidget(widget)

    def add_spacing(self, px: int = 4):
        self._layout.addSpacing(px)


class LogWidget(QTextEdit):
    """Auto-scrolling log panel"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(
            f"background:{PAL['bg0']}; color:{PAL['text2']};"
            f"border:1px solid {PAL['border2']}; font-family:monospace; font-size:11px;")

    def append_log(self, text: str):
        self.append(text)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


class MetricsBadge(QWidget):
    """Compact coloured badge showing IoU/Dice/Prec/Rec for one image"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        hl = QHBoxLayout(self)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(8)
        self._labels = {}
        for key, colour in [("IoU", "#42A5F5"), ("Dice", "#66BB6A"),
                             ("Prec", "#FF9800"), ("Rec", "#E91E63")]:
            lbl = QLabel(f"{key}: —")
            lbl.setStyleSheet(
                f"color:{colour}; font-weight:700; font-size:12px;"
                f"background:{PAL['bg3']}; padding:2px 8px; border-radius:4px;")
            hl.addWidget(lbl)
            self._labels[key] = lbl
        hl.addStretch()

    def update_metrics(self, metrics: dict):
        mapping = {"IoU": "IoU", "Dice": "Dice", "Precision": "Prec", "Recall": "Rec"}
        for k, short in mapping.items():
            if k in metrics:
                self._labels[short].setText(f"{short}: {metrics[k]:.4f}")
