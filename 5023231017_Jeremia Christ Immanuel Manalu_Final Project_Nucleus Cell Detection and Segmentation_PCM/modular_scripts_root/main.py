#!/usr/bin/env python3
"""
modular_scripts_root: MoNuSeg 2018 Nuclei Segmentation GUI
Usage:
    python -m modular_scripts_root          (from the parent folder)
    python main.py                (from inside the modular_scripts_root folder)

Optionally place background.jpg/background.png in the project root
to enable the custom background feature
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor, QPalette

from modular_scripts_root.ui.theme       import PAL, DARK_QSS
from modular_scripts_root.ui.main_window import NucleiSegApp


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(PAL["bg1"]))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(PAL["text0"]))
    pal.setColor(QPalette.ColorRole.Base,            QColor(PAL["bg4"]))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(PAL["bg3"]))
    pal.setColor(QPalette.ColorRole.Text,            QColor(PAL["text0"]))
    pal.setColor(QPalette.ColorRole.Button,          QColor(PAL["bg3"]))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(PAL["text0"]))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(PAL["accent1"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(PAL["text0"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(PAL["bg3"]))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(PAL["text0"]))
    app.setPalette(pal)

    win = NucleiSegApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
