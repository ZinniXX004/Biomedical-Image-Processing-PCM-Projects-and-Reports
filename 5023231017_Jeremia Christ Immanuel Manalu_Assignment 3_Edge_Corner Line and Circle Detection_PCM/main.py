import sys
import matplotlib
matplotlib.use("QtAgg")  # Set the backend before importing any plotting components
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    # Initialize the PyQt6 Application
    app = QApplication(sys.argv)
    
    # Set global style for a modern look
    app.setStyle("Fusion")
    
    # Instantiate and show the main window
    window = MainWindow()
    window.show()
    
    # Execute the application loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()