import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from PyQt6.QtGui import QFont, QColor, QPainter, QPixmap, QImage
from PyQt6.QtCore import Qt

from config import (
    BG_DARK, BG_PANEL, BG_CARD, TEXT_COL, SUBTEXT, GRID_COL,
    STAGE_COLORS, ENH_COLORS
)

class ImagePanel(QWidget):
    def __init__(self, title: str, subtitle: str, stage_category: str, parent=None):
        super().__init__(parent)
        self._colorbar = None
        self._border = STAGE_COLORS.get(stage_category, GRID_COL)
        self.setObjectName("ImagePanel")
        self.setStyleSheet(f"#ImagePanel {{ border: 1px solid {GRID_COL}; border-radius: 8px; background-color: {BG_CARD}; }}")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        hdr = QWidget()
        hdr.setObjectName("panelHdr")
        hdr.setStyleSheet(f"#panelHdr {{ background-color: {BG_PANEL}; border-radius: 6px; }}")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 8, 10, 8)
        
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{self._border}; font-size:10px; background:transparent; border:none;")
        t_lbl = QLabel(title)
        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        t_lbl.setStyleSheet(f"color:{TEXT_COL}; background:transparent; border:none;")
        s_lbl = QLabel(subtitle)
        s_lbl.setFont(QFont("Segoe UI", 9))
        s_lbl.setStyleSheet(f"color:{SUBTEXT}; background:transparent; border:none;")

        hl.addWidget(dot)
        hl.addWidget(t_lbl)
        hl.addStretch()
        hl.addWidget(s_lbl)
        root.addWidget(hdr)

        self.fig = Figure(figsize=(4.0, 3.2), facecolor=BG_CARD)
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        self.ax = self.fig.add_subplot(111)
        self._style_ax()

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color:{BG_CARD}; border:none; border-radius:6px;")
        
        self.toolbar = NavToolbar(self.canvas, self)
        self.toolbar.setMaximumHeight(30)
        self.toolbar.setStyleSheet(
            f"QToolBar {{ background:{BG_PANEL}; border:none; padding: 2px; }}"
            f"QToolButton {{ background:{BG_DARK}; border:1px solid {GRID_COL}; border-radius:4px; color:{TEXT_COL}; padding:4px; min-width:24px; }}"
            f"QToolButton:hover {{ background:{GRID_COL}; border-color:{self._border}; }}"
        )

        root.addWidget(self.canvas, 1)
        root.addWidget(self.toolbar)
        self.setMinimumSize(340, 300)

    def _style_ax(self):
        self.ax.set_facecolor(BG_PANEL)
        for sp in self.ax.spines.values():
            sp.set_color(GRID_COL)
        self.ax.tick_params(colors=SUBTEXT, labelsize=8)

    def _reset_ax(self):
        if self._colorbar is not None:
            try: self._colorbar.remove()
            except Exception: pass
            self._colorbar = None
        self.ax.clear()
        self._style_ax()

    def show_image(self, img, cmap="gray", vmin=0.0, vmax=1.0, colorbar=False):
        self._reset_ax()
        im = self.ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear", aspect="equal")
        self.ax.axis("off")
        if colorbar:
            self._colorbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.02)
            self._colorbar.ax.tick_params(colors=SUBTEXT, labelsize=8)
            for sp in self._colorbar.ax.spines.values():
                sp.set_color(GRID_COL)
        self.canvas.draw_idle()

    def show_histogram(self, nm, thr, method_color):
        self._reset_ax()
        hist, bins = np.histogram(nm.ravel(), bins=80, range=(0, 1))
        self.ax.fill_between(bins[:-1], 0, hist, alpha=0.35, color=method_color)
        self.ax.plot(bins[:-1], hist, color=method_color, linewidth=1.5)
        self.ax.axvline(thr, color="white", linestyle="--", linewidth=1.5, label=f"thr={thr:.2f}")
        self.ax.legend(fontsize=8, loc="upper right", facecolor=BG_PANEL, edgecolor=GRID_COL, labelcolor=TEXT_COL)
        self.ax.grid(True, alpha=0.15, color=SUBTEXT)
        self.fig.subplots_adjust(left=0.15, right=0.92, top=0.92, bottom=0.18)
        self.canvas.draw_idle()

    def show_profile(self, nm, binary, thr, method_color, row):
        self._reset_ax()
        self.ax.plot(nm[row, :], color=method_color, lw=1.8, label="Magnitude")
        self.ax.plot(binary[row, :], color="white", lw=1.0, alpha=0.5, label="Binary")
        self.ax.axhline(thr, color="yellow", lw=1.2, linestyle=":", label=f"thr={thr:.2f}")
        self.ax.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COL, labelcolor=TEXT_COL)
        self.ax.grid(True, alpha=0.15, color=SUBTEXT)
        self.ax.set_ylim(-0.05, 1.1)
        self.fig.subplots_adjust(left=0.15, right=0.92, top=0.92, bottom=0.18)
        self.canvas.draw_idle()

    def save_to(self, filepath: str) -> bool:
        try:
            self.fig.savefig(filepath, dpi=120, bbox_inches="tight", facecolor=BG_CARD, edgecolor="none")
            return True
        except Exception:
            return False

class AnalysisPanel(QWidget):
    def __init__(self, title: str, subtitle: str, stage_category: str, figsize: tuple = (4.0, 3.2), parent=None):
        super().__init__(parent)
        self._border = STAGE_COLORS.get(stage_category, GRID_COL)
        self.setObjectName("AnalysisPanel")
        self.setStyleSheet(f"#AnalysisPanel {{ border: 1px solid {GRID_COL}; border-radius: 8px; background-color: {BG_CARD}; }}")
        
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        hdr = QWidget()
        hdr.setObjectName("apHdr")
        hdr.setStyleSheet(f"#apHdr {{ background-color: {BG_PANEL}; border-radius: 6px; }}")
        hl = QHBoxLayout(hdr)
        
        dot = QLabel("◈")
        dot.setStyleSheet(f"color:{self._border}; font-size:12px; background:transparent; border:none;")
        t_lbl = QLabel(title)
        t_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        t_lbl.setStyleSheet(f"color:{TEXT_COL}; background:transparent; border:none;")
        s_lbl = QLabel(subtitle)
        s_lbl.setFont(QFont("Segoe UI", 9))
        s_lbl.setStyleSheet(f"color:{SUBTEXT}; background:transparent; border:none;")

        hl.addWidget(dot); hl.addWidget(t_lbl); hl.addStretch(); hl.addWidget(s_lbl)
        root.addWidget(hdr)

        self.fig = Figure(figsize=figsize, facecolor=BG_CARD)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color:{BG_CARD}; border:none; border-radius:6px;")
        
        root.addWidget(self.canvas, 1)
        self.setMinimumSize(340, 300)

    def _fresh_ax(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(BG_PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID_COL)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        return ax

    @staticmethod
    def _style_twin(ax2):
        for sp in ax2.spines.values():
            sp.set_color(GRID_COL)
        ax2.tick_params(colors=SUBTEXT, labelsize=8)

    def show_hist_ogive(self, img: np.ndarray, label: str = "", color: str = "#89B4FA") -> None:
        ax = self._fresh_ax()
        hist, bins = np.histogram(img.ravel(), bins=80, range=(0.0, 1.0))
        cdf = np.cumsum(hist) / img.size * 100.0
        centers = (bins[:-1] + bins[1:]) / 2.0
        bw = centers[1] - centers[0]

        ax.bar(centers, hist, width=bw, color=color, alpha=0.55)
        ax.plot(centers, hist, color=color, lw=1.5, label="Histogram")
        ax.grid(True, alpha=0.15, color=SUBTEXT)

        ax2 = ax.twinx()
        self._style_twin(ax2)
        ax2.plot(centers, cdf, color="white", lw=2.0, linestyle="--", alpha=0.9, label="Ogive (CDF %)")
        ax2.set_ylim(0.0, 105.0)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COL, labelcolor=TEXT_COL, loc="upper left")
        if label: ax.set_title(label, fontsize=10, color=TEXT_COL, pad=6)

        self.fig.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.18)
        self.canvas.draw_idle()

    def show_comparison_hist_ogive(self, images_dict: dict, title: str = "") -> None:
        _cycle = ["#89B4FA", "#FAB387", "#A6E3A1", "#CBA6F7"]
        ax = self._fresh_ax()
        ax2 = ax.twinx()
        self._style_twin(ax2)

        for i, (lbl, img) in enumerate(images_dict.items()):
            c = _cycle[i % len(_cycle)]
            hist, bins = np.histogram(img.ravel(), bins=80, range=(0.0, 1.0))
            cdf = np.cumsum(hist) / img.size * 100.0
            centers = (bins[:-1] + bins[1:]) / 2.0
            ax.plot(centers, hist, color=c, lw=1.8, alpha=0.85, label=f"{lbl} Hist")
            ax2.plot(centers, cdf, color=c, lw=1.8, linestyle="--", alpha=0.85, label=f"{lbl} CDF")

        ax2.set_ylim(0.0, 105.0)
        ax.grid(True, alpha=0.15, color=SUBTEXT)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COL, labelcolor=TEXT_COL, loc="upper left", ncol=2)
        if title: ax.set_title(title, fontsize=10, color=TEXT_COL, pad=6)

        self.fig.subplots_adjust(left=0.18, right=0.82, top=0.88, bottom=0.18)
        self.canvas.draw_idle()

    def show_performance_eval(self, metrics_per_method: dict) -> None:
        self.fig.clear()
        metric_cfg = [
            ("RMSE",    "↓ lower = better",          "#F38BA8"),
            ("PSNR",    "↑ higher = better [dB]",    "#A6E3A1"),
            ("SSIM",    "↑ higher = better [0–1]",   "#89B4FA"),
            ("Entropy", "↑ higher = info [bits]",    "#CBA6F7"),
        ]
        methods = list(metrics_per_method.keys())
        bar_clrs = [ENH_COLORS.get(m, "#89B4FA") for m in methods]

        for idx, (mname, hint, accent) in enumerate(metric_cfg):
            ax = self.fig.add_subplot(2, 2, idx + 1)
            ax.set_facecolor(BG_PANEL)
            for sp in ax.spines.values(): sp.set_color(GRID_COL)
            ax.tick_params(colors=SUBTEXT, labelsize=8)

            vals = [metrics_per_method[m].get(mname, 0.0) for m in methods]
            bars = ax.bar(methods, vals, color=bar_clrs, alpha=0.85, width=0.5, edgecolor=GRID_COL)

            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_COL)

            ax.set_title(f"{mname} ({hint})", fontsize=10, color=accent, pad=6)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, fontsize=9, color=TEXT_COL, fontweight="bold")
            ax.grid(True, alpha=0.15, color=SUBTEXT, axis="y")

        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, hspace=0.65, wspace=0.35)
        self.canvas.draw_idle()

    def show_runtime_bars(self, runtimes_dict: dict) -> None:
        ax = self._fresh_ax()
        methods = list(runtimes_dict.keys())
        times = [runtimes_dict[m] for m in methods]
        t_max = max(times) + 1e-9

        bar_colors =[]
        for t in times:
            ratio = t / t_max
            if ratio < 0.25:   bar_colors.append("#A6E3A1")
            elif ratio < 0.60: bar_colors.append("#E3A968")
            else:              bar_colors.append("#F38BA8")

        bars = ax.barh(methods, times, color=bar_colors, alpha=0.85, edgecolor=GRID_COL, height=0.55)
        for bar, t in zip(bars, times):
            ax.text(bar.get_width() + t_max * 0.012, bar.get_y() + bar.get_height() / 2.0,
                    f"{t:.1f} ms", va="center", ha="left", fontsize=9, color=TEXT_COL)

        ax.set_title("Method Runtime Comparison  ·  (green) fast   (orange) medium  (red) slow", fontsize=11, color=TEXT_COL, pad=8)
        ax.grid(True, alpha=0.15, color=SUBTEXT, axis="x")
        ax.set_xlim(0, t_max * 1.32)
        self.fig.subplots_adjust(left=0.22, right=0.88, top=0.90, bottom=0.12)
        self.canvas.draw_idle()

    def show_image_grid(self, images_dict: dict, title: str = "", cmap: str = "gray", cols: int = 4) -> None:
        self.fig.clear()
        n = len(images_dict)
        rows = max(1, (n + cols - 1) // cols)

        for idx, (lbl, img) in enumerate(images_dict.items()):
            ax = self.fig.add_subplot(rows, cols, idx + 1)
            ax.set_facecolor(BG_PANEL)
            if img is not None:
                if img.ndim == 3: ax.imshow(img.astype(np.uint8), aspect="equal", interpolation="bilinear")
                else: ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal", interpolation="bilinear")
            ax.set_title(lbl, fontsize=8, color=TEXT_COL, pad=3)
            ax.axis("off")

        if title: self.fig.suptitle(title, fontsize=11, color=TEXT_COL, y=0.99)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02, hspace=0.38, wspace=0.06)
        self.canvas.draw_idle()

    def show_rgb_image(self, rgb_img: np.ndarray, title: str = "") -> None:
        ax = self._fresh_ax()
        ax.imshow(rgb_img.astype(np.uint8), aspect="equal", interpolation="bilinear")
        ax.axis("off")
        if title: ax.set_title(title, fontsize=10, color=TEXT_COL, pad=6)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
        self.canvas.draw_idle()

    # --- FUNGSI SAVE DITAMBAHKAN DI SINI ---
    def save_to(self, filepath: str) -> bool:
        try:
            self.fig.savefig(filepath, dpi=120, bbox_inches="tight", facecolor=BG_CARD, edgecolor="none")
            return True
        except Exception as e:
            print(f"[SAVE ERROR] {filepath}: {e}")
            return False

class BackgroundWidget(QWidget):
    def __init__(self, image_path: str, opacity: float = 0.3, dimness: float = 0.6, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.opacity = opacity
        self.dimness = dimness
        self._original_image = QImage(self.image_path)
        self._cached_pixmap = None
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_background_cache()

    def _update_background_cache(self):
        if self._original_image.isNull() or self.width() == 0 or self.height() == 0: return
        scaled_img = self._original_image.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        self._cached_pixmap = QPixmap.fromImage(scaled_img)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG_DARK))
        
        if self._cached_pixmap and not self._cached_pixmap.isNull():
            painter.setOpacity(self.opacity)
            x = (self.width() - self._cached_pixmap.width()) // 2
            y = (self.height() - self._cached_pixmap.height()) // 2
            painter.drawPixmap(x, y, self._cached_pixmap)
        
        if self.dimness > 0:
            painter.setOpacity(self.dimness)
            painter.fillRect(self.rect(), QColor(BG_DARK))