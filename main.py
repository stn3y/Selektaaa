import sys
import logging
from PyQt5.QtWidgets import QApplication
from gui_playlist_generator import PlaylistGeneratorGUI

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create the application and GUI
    app = QApplication(sys.argv)
    gui = PlaylistGeneratorGUI()

    # Show the GUI
    gui.show()

    # Start the application event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()