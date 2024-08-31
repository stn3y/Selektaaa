import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import csv
import yaml
import logging
from src.dqn_agent import DQNAgent
from src.playlist_environment import PlaylistEnvironment
from src.feature_extraction import analyze_tracks

class PlaylistGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Playlist Generator")
        self.tracks = None
        self.config = self.load_config()

        self.create_reward_weight_slider("Similarity Weight", 0, 1.0)
        self.create_reward_weight_slider("Tempo Weight", 1, 1.0)
        self.create_reward_weight_slider("Energy Weight", 2, 1.0)

        ttk.Label(self.root, text="Crossfade Duration (seconds)").grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
        self.crossfade_duration_slider = ttk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL)
        self.crossfade_duration_slider.set(5)
        self.crossfade_duration_slider.grid(row=5, column=1, padx=10, pady=5)

        analyze_button = ttk.Button(root, text="Analyze Tracks", command=self.analyze_tracks)
        analyze_button.grid(row=6, columnspan=2, pady=10)

        self.progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.grid(row=7, columnspan=2, pady=10)

        generate_button = ttk.Button(root, text="Generate Playlist", command=self.generate_playlist)
        generate_button.grid(row=8, columnspan=2, pady=10)

        feedback_button = ttk.Button(root, text="Provide Feedback", command=self.provide_feedback)
        feedback_button.grid(row=9, columnspan=2, pady=10)

        self.status_label = ttk.Label(root, text="Status: Waiting for user action")
        self.status_label.grid(row=10, columnspan=2, pady=5)

    def load_config(self):
        try:
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {e}")
            self.root.quit()

    def create_reward_weight_slider(self, label_text, row, default_value):
        label = ttk.Label(self.root, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=5, sticky=tk.W)
        
        slider = ttk.Scale(self.root, from_=0.0, to=10.0, orient=tk.HORIZONTAL)
        slider.set(default_value)
        slider.grid(row=row, column=1, padx=10, pady=5)

        setattr(self, f"{label_text.lower().replace(' ', '_')}_slider", slider)

    def analyze_tracks(self):
        directory = filedialog.askdirectory(title="Select Directory Containing Tracks")
        logging.info(f"Selected directory: {directory}")
        if directory:
            self.status_label.config(text="Status: Analyzing tracks...")
            self.progress_bar['value'] = 0
            self.root.update_idletasks()

            try:
                self.tracks = analyze_tracks(directory, progress_callback=self.update_progress)
                logging.info(f"Extracted tracks: {self.tracks}")
                if self.tracks:
                    self.status_label.config(text="Status: Tracks loaded successfully!")
                else:
                    self.status_label.config(text="Error: No valid tracks found.")
            except Exception as e:
                self.status_label.config(text=f"Error loading tracks: {e}")
                logging.error(f"Failed to analyze tracks: {e}")
                messagebox.showerror("Error", f"Failed to analyze tracks: {e}")

    def update_progress(self, value):
        self.progress_bar['value'] = value
        self.root.update_idletasks()

    def generate_playlist(self):
        if self.tracks is None or len(self.tracks) == 0:
            self.status_label.config(text="Error: No tracks loaded. Please load tracks first.")
            messagebox.showerror("Error", "No tracks loaded. Please load tracks first.")
            return

        reward_weights = {
            'similarity': self.similarity_weight_slider.get(),
            'tempo': self.tempo_weight_slider.get(),
            'energy': self.energy_weight_slider.get(),
        }
        crossfade_duration = self.crossfade_duration_slider.get()

        try:
            self.status_label.config(text="Generating playlist...")
            self.root.update_idletasks()

            env = PlaylistEnvironment(self.tracks, reward_weights=reward_weights)
            agent = DQNAgent(env.feature_dim, len(self.tracks), self.config)

            logging.info("Starting playlist generation...")
            playlist = agent.generate_playlist(env)

            if not playlist:
                raise ValueError("Generated playlist is empty.")
            
            logging.info(f"Generated playlist with {len(playlist)} tracks.")
            self.save_playlist_to_csv(playlist)

            self.status_label.config(text="Playlist generated successfully!")
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")
            logging.error(f"Failed to generate playlist: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to generate playlist: {e}")

    def save_playlist_to_csv(self, playlist):
        csv_filename = f'generated_playlist_{len(os.listdir())}.csv'
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Track Filename'])
            for track in playlist:
                writer.writerow([track['filename']])
        messagebox.showinfo("Playlist Saved", f"Playlist has been saved to {csv_filename}")

    def provide_feedback(self):
        feedback = messagebox.askyesno("Feedback", "Are you satisfied with the playlist?")
        if feedback:
            self.status_label.config(text="Positive feedback received. Adjusting model...")
        else:
            self.status_label.config(text="Negative feedback received. Retraining model...")

        try:
            self.generate_playlist()
        except Exception as e:
            self.status_label.config(text=f"Error during feedback processing: {e}")
            logging.error(f"Failed during feedback processing: {e}")
            messagebox.showerror("Error", f"Failed during feedback processing: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    gui = PlaylistGeneratorGUI(root)
    root.mainloop()