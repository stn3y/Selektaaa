import sys
import csv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, 
    QFileDialog, QProgressBar, QLineEdit, QMessageBox, QTextEdit, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import logging
from src.feature_extraction import analyze_tracks
from src.playlist_environment import PlaylistEnvironment
from src.dqn_agent import DQNAgent

class TrackAnalysisThread(QThread):
    progress = pyqtSignal(int)
    analysis_done = pyqtSignal(list)

    def __init__(self, directory, csv_file_path):
        super().__init__()
        self.directory = directory
        self.csv_file_path = csv_file_path

    def run(self):
        tracks = analyze_tracks(self.directory, self.csv_file_path, progress_callback=self.emit_progress)
        self.analysis_done.emit(tracks)

    def emit_progress(self, value):
        self.progress.emit(int(value))

class PlaylistGeneratorGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.tracks = None
        self.env = None

        # Initialize reward weights
        self.reward_weights = {
            'similarity': 1.0,
            'tempo': 0.2,
            'key': 0.2,
            'energy': 0.2,
            'length': 0.2,
            'dynamic_range': 0.2,
            'harmonicity': 0.2,
            'onset_strength': 0.2,
            'rhythm': 0.2
        }

        # Set up the main window
        self.setWindowTitle("AI DJ Playlist Generator")
        self.setGeometry(100, 100, 800, 600)

        # Set up central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Title label
        self.title_label = QLabel("AI DJ Playlist Generator", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.title_label)

        # Reward weight sliders
        self.reward_weight_sliders = {}
        self.add_reward_weight_slider('similarity', "Similarity Weight", 1.0)
        self.add_reward_weight_slider('tempo', "Tempo Weight", 0.2)
        self.add_reward_weight_slider('key', "Key Weight", 0.2)
        self.add_reward_weight_slider('energy', "Energy Weight", 0.2)
        self.add_reward_weight_slider('length', "Length Weight", 0.2)
        self.add_reward_weight_slider('dynamic_range', "Dynamic Range Weight", 0.2)
        self.add_reward_weight_slider('harmonicity', "Harmonicity Weight", 0.2)
        self.add_reward_weight_slider('onset_strength', "Onset Strength Weight", 0.2)
        self.add_reward_weight_slider('rhythm', "Rhythm Weight", 0.2)

        # Button to select and analyze files
        self.analyze_button = QPushButton("Select and Analyze Files", self)
        self.analyze_button.clicked.connect(self.analyze_tracks)
        self.layout.addWidget(self.analyze_button)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        # Button to generate playlist
        self.generate_button = QPushButton("Generate Playlist", self)
        self.generate_button.clicked.connect(self.generate_playlist)
        self.generate_button.setEnabled(False)
        self.layout.addWidget(self.generate_button)

        # Feedback entry
        self.feedback_label = QLabel("Provide Feedback (0-5):", self)
        self.layout.addWidget(self.feedback_label)
        self.feedback_entry = QLineEdit(self)
        self.layout.addWidget(self.feedback_entry)

        # Log display
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.layout.addWidget(self.log_text)

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", handlers=[LoggingHandler(self.log_text)])

    def add_reward_weight_slider(self, key, label, default_value):
        slider_label = QLabel(f"{label}:", self)
        self.layout.addWidget(slider_label)

        slider = QSlider(Qt.Horizontal, self)
        slider.setRange(0, 100)
        slider.setValue(int(default_value * 100))
        slider.valueChanged.connect(lambda value, k=key: self.update_reward_weight(k, value))
        self.layout.addWidget(slider)

        self.reward_weight_sliders[key] = slider

    def update_reward_weight(self, key, value):
        self.reward_weights[key] = value / 100.0
        logging.info(f"Updated {key} weight to {value / 100.0}")

    def analyze_tracks(self):
        # Open file dialogs for directory and CSV file
        directory = QFileDialog.getExistingDirectory(self, "Select Directory Containing Audio Files")
        if not directory:
            QMessageBox.warning(self, "No Directory Selected", "Please select a directory.")
            return

        csv_file_path, _ = QFileDialog.getOpenFileName(self, "Select Mixed In Key CSV File", "", "CSV Files (*.csv)")
        if not csv_file_path:
            QMessageBox.warning(self, "No CSV File Selected", "Please select a Mixed In Key CSV file.")
            return

        self.progress_bar.setValue(0)
        self.track_analysis_thread = TrackAnalysisThread(directory, csv_file_path)
        self.track_analysis_thread.progress.connect(self.update_progress)
        self.track_analysis_thread.analysis_done.connect(self.on_analysis_done)
        self.track_analysis_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def on_analysis_done(self, tracks):
        if not tracks:
            QMessageBox.critical(self, "Analysis Failed", "No tracks were successfully analyzed. Please try again.")
            return

        self.tracks = tracks
        self.env = PlaylistEnvironment(self.tracks, reward_weights=self.reward_weights)
        config = {
            "gamma": 0.99,
            "learning_rate": 0.001
        }
        self.agent = DQNAgent(self.env.feature_dim, len(self.tracks), config)

        self.generate_button.setEnabled(True)
        QMessageBox.information(self, "Analysis Complete", "Tracks analyzed successfully!")

    def generate_playlist(self):
        if not self.tracks:
            QMessageBox.critical(self, "Error", "No tracks loaded. Please analyze tracks first.")
            return

        playlist = self.agent.generate_playlist(self.env)

        feedback = self.collect_feedback()
        self.env.apply_user_feedback(feedback)

        output_path, _ = QFileDialog.getSaveFileName(self, "Save Playlist", "", "CSV Files (*.csv)")
        if output_path:
            self.save_playlist_to_csv(playlist, output_path)
            QMessageBox.information(self, "Success", f"Playlist saved to {output_path}")
        else:
            QMessageBox.warning(self, "Save Cancelled", "Playlist was not saved.")

    def save_playlist_to_csv(self, playlist, output_path):
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Track Filename'])
            for track in playlist:
                writer.writerow([track['filename']])

    def collect_feedback(self):
        try:
            feedback = float(self.feedback_entry.text())
            if 0 <= feedback <= 5:
                return {"overall_rating": feedback}
            else:
                raise ValueError
        except ValueError:
            QMessageBox.critical(self, "Invalid Feedback", "Please enter a number between 0 and 5.")
            return {}

class LoggingHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.append(msg)
        self.text_widget.ensureCursorVisible()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = PlaylistGeneratorGUI()
    gui.show()
    sys.exit(app.exec_())