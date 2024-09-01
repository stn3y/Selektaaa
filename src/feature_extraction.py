import os
import re
import librosa
from librosa.feature import rhythm
import numpy as np
import csv
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, filedialog

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_filename(filename):
    """Normalize the filename by removing non-alphanumeric characters and converting to lowercase."""
    filename = filename.lower()
    filename = re.sub(r'[^a-z0-9]', '', filename)
    return filename

def update_track_metadata(tracks, csv_file_path):
    """
    Update the track metadata with key, tempo, and energy from Mixed In Key CSV file.

    Args:
        tracks (list): A list of track dictionaries, each containing attributes like 'filename'.
        csv_file_path (str): Path to the CSV file exported from Mixed In Key.
    
    Returns:
        list: Updated list of track dictionaries with key, tempo, and energy attributes.
    """
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        mixed_in_key_data = list(reader)

    for track in tracks:
        normalized_name = normalize_filename(os.path.splitext(track['filename'])[0])

        # Search for a matching row in the CSV file
        matching_metadata = next(
            (row for row in mixed_in_key_data if normalize_filename(row['File name']) == normalized_name),
            None
        )

        if matching_metadata:
            track['key'] = matching_metadata.get('Key result')
            track['tempo'] = float(matching_metadata.get('BPM', track.get('tempo', 0)))
            track['energy'] = int(matching_metadata.get('Energy', track.get('energy', 0)))
        else:
            logging.warning(f"No metadata found for {track['filename']}")

    return tracks

def extract_librosa_features(y, sr):
    """
    Extract a comprehensive set of audio features using librosa.

    Args:
        y (numpy array): Audio time series.
        sr (int): Sampling rate of y.
    
    Returns:
        tuple: Combined and flattened feature array, along with tempo and beat frames.
    """
    # Extract features
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).flatten()
    mfcc = librosa.feature.mfcc(y=y, sr=sr).flatten()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).flatten()
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).flatten()
    zcr = librosa.feature.zero_crossing_rate(y).flatten()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).flatten()
    rms = librosa.feature.rms(y=y).flatten()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()

    # Additional features
    harmonicity = librosa.effects.harmonic(y).flatten()  # Harmonicity
    onset_env = librosa.onset.onset_strength(y=y, sr=sr).flatten()  # Onset detection
    tempo_rhythm = rhythm.tempo(y=y, sr=sr).flatten()  # Rhythmic patterns

    # Combine all features
    combined_features = np.hstack([
        chroma, mfcc, spectral_contrast, tonnetz,
        zcr, spectral_rolloff, rms, spectral_centroid,
        harmonicity, onset_env, tempo_rhythm
    ])

    return combined_features, tempo, beat_frames

def analyze_tracks_with_librosa(tracks, target_length=1024):
    """
    Analyze tracks using librosa to extract features like chroma, mfcc, and beat positions.

    Args:
        tracks (list): A list of track dictionaries, each containing a 'filename' attribute.
        target_length (int): The length to pad or truncate features to.

    Returns:
        list: Updated list of track dictionaries with additional features extracted by librosa.
    """
    scaler = StandardScaler()
    all_features = []

    for track in tqdm(tracks, desc="Analyzing Tracks with Librosa"):
        try:
            y, sr = librosa.load(track['filename'], sr=None)

            # Extract features, track length, and dynamic range
            combined_features, tempo, beat_frames = extract_librosa_features(y, sr)

            # Log track length and dynamics
            track_length = librosa.get_duration(y=y, sr=sr)
            dynamic_range = np.max(y) - np.min(y)

            # Ensure consistent feature length
            track['features'] = pad_or_truncate(combined_features, target_length)
            track['length'] = track_length
            track['dynamic_range'] = dynamic_range
            track['tempo'] = tempo
            track['beat_frames'] = beat_frames
            all_features.append(track['features'])

        except Exception as e:
            logging.error(f"Error processing {track['filename']}: {e}")

    # Normalize features
    all_features = scaler.fit_transform(all_features)
    for idx, track in enumerate(tracks):
        track['features'] = all_features[idx]

    return tracks

def pad_or_truncate(features, target_length):
    """
    Pad or truncate the feature vector to ensure consistent length.

    Args:
        features (numpy array): The feature array to be padded or truncated.
        target_length (int): The desired length of the feature array.

    Returns:
        numpy array: The padded or truncated feature array.
    """
    if len(features) > target_length:
        return features[:target_length]
    elif len(features) < target_length:
        return np.pad(features, (0, target_length - len(features)), 'constant')
    return features

def analyze_tracks(directory, csv_file_path, progress_callback=None):
    """
    Main function to analyze tracks in a directory and update with metadata from a CSV.

    Args:
        directory (str): Path to the directory containing audio files.
        csv_file_path (str): Path to the Mixed In Key CSV file.
        progress_callback (function): Optional function to update progress.
    
    Returns:
        list: List of tracks with extracted features and metadata.
    """
    tracks = []
    audio_files = [f for f in os.listdir(directory) if f.endswith('.mp3') or f.endswith('.wav')]
    total_files = len(audio_files)

    for idx, filename in enumerate(audio_files):
        file_path = os.path.join(directory, filename)
        try:
            # Load the audio file
            y, sr = librosa.load(file_path, sr=None)

            # Extract features
            combined_features, tempo, beat_frames = extract_librosa_features(y, sr)

            # Log track length and dynamics
            track_length = librosa.get_duration(y=y, sr=sr)
            dynamic_range = np.max(y) - np.min(y)

            # Ensure consistent feature length
            combined_features = pad_or_truncate(combined_features, target_length=1024)

            track = {
                'filename': filename,
                'features': combined_features,
                'tempo': tempo,
                'beat_frames': beat_frames,
                'length': track_length,
                'dynamic_range': dynamic_range
            }
            tracks.append(track)

            # Update progress
            if progress_callback:
                progress_callback((idx + 1) / total_files * 100)

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

    # Update tracks with additional metadata from Mixed In Key CSV
    if csv_file_path:
        tracks = update_track_metadata(tracks, csv_file_path)

    return tracks

def prompt_for_files():
    """
    Prompt the user to select the directory containing audio files and the Mixed In Key CSV file.

    Returns:
        tuple: A tuple containing the selected directory path and the CSV file path.
    """
    root = Tk()
    root.withdraw()  # Hide the root window

    directory = filedialog.askdirectory(title="Select Directory Containing Audio Files")
    if not directory:
        logging.error("No directory selected. Exiting.")
        return None, None

    csv_file_path = filedialog.askopenfilename(
        title="Select Mixed In Key CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )
    if not csv_file_path:
        logging.error("No CSV file selected. Exiting.")
        return None, None

    return directory, csv_file_path