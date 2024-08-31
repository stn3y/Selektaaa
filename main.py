import logging
from gui_playlist_generator import PlaylistGeneratorGUI
import tkinter as tk

def main():
    logging.basicConfig(
        filename='playlist_generator.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("Starting Playlist Generator Application")

    try:
        root = tk.Tk()
        gui = PlaylistGeneratorGUI(root)
        root.mainloop()
    except Exception as e:
        logging.critical("An unhandled exception occurred", exc_info=True)
        raise e
    finally:
        logging.info("Playlist Generator Application has exited")

if __name__ == "__main__":
    main()