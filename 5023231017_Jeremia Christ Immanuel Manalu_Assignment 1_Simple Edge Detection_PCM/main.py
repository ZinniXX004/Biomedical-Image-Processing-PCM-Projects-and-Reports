import sys
import matplotlib
# Must be set before importing any PyQt6 plotting components
matplotlib.use("QtAgg")

from PyQt6.QtWidgets import QApplication
from main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Modern cross-platform style
    app.setStyle("Fusion")
    
    # Launch main window
    win = MainWindow()
    win.show()
    
    sys.exit(app.exec())