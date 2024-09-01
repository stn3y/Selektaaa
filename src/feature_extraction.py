import os
import re
import librosa
import numpy as np
import csv
import logging
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tkinter import Tk, filedialog
from rapidfuzz import fuzz, process  # Ensure this import is included

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_filename_based_on_csv_patterns(filename, csv_patterns):
    filename = filename.lower()
    if csv_patterns.get("replace_underscores"):
        filename = filename.replace('_', ' ')
    if csv_patterns.get("replace_hyphens"):
        filename = filename.replace('-', ' ')
    if csv_patterns.get("remove_parentheses"):
        filename = re.sub(r'\(.*?\)|\[.*?\]', '', filename)
    if csv_patterns.get("remove_common_terms"):
        filename = re.sub(r'\b(ft\.?|feat\.?|remix|mix|edit)\b', '', filename)
    filename = re.sub(r'[^a-z0-9\s]', '', filename)
    filename = re.sub(r'\s+', ' ', filename).strip()
    return filename

def extract_csv_formatting_patterns(csv_file_path):
    patterns = {"replace_underscores": False, "replace_hyphens": False, "remove_parentheses": False, "remove_common_terms": False}
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['File name'].lower()
            if '_' in filename:
                patterns["replace_underscores"] = True
            if '-' in filename:
                patterns["replace_hyphens"] = True
            if re.search(r'\(.*?\)|\[.*?\]', filename):
                patterns["remove_parentheses"] = True
            if re.search(r'\b(ft\.?|feat\.?|remix|mix|edit)\b', filename):
                patterns["remove_common_terms"] = True
    return patterns

def fuzzy_match_track(normalized_name, csv_data, threshold=80):
    """
    Fuzzy match the normalized track name with entries in the CSV file using rapidfuzz.

    Args:
        normalized_name (str): Normalized track name.
        csv_data (list): List of dictionaries representing CSV rows.
        threshold (int): Similarity threshold for considering a match.

    Returns:
        dict or None: The best match from the CSV file, or None if no match is found above the threshold.
    """
    csv_filenames = [normalize_filename_based_on_csv_patterns(row['File name'], {}) for row in csv_data]
    
    # Using underscore to ignore additional return values
    best_match, score, *_ = process.extractOne(normalized_name, csv_filenames, scorer=fuzz.token_sort_ratio)

    if score >= threshold:
        return csv_data[csv_filenames.index(best_match)]
    return None


def update_track_metadata(tracks, csv_file_path):
    """Update the track metadata with key, tempo, and energy from the Mixed In Key CSV file."""
    csv_patterns = extract_csv_formatting_patterns(csv_file_path)

    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        mixed_in_key_data = list(reader)

    unmatched_tracks = []

    for track in tracks:
        normalized_name = normalize_filename_based_on_csv_patterns(os.path.splitext(track['filename'])[0], csv_patterns)
        artist, title = extract_artist_and_title(normalized_name)

        possible_filenames = [normalize_filename_based_on_csv_patterns(row['File name'], csv_patterns) for row in mixed_in_key_data]

        matching_metadata = fuzzy_match_track(normalized_name, mixed_in_key_data)

        if not matching_metadata:
            best_filename_match, filename_score, *_ = process.extractOne(normalized_name, possible_filenames, scorer=fuzz.token_sort_ratio)

            if filename_score >= 80:
                matching_metadata = mixed_in_key_data[possible_filenames.index(best_filename_match)]

        if matching_metadata:
            track['key'] = matching_metadata.get('Key result')
            track['tempo'] = float(matching_metadata.get('BPM', track.get('tempo', 0)))
            track['energy'] = int(matching_metadata.get('Energy', track.get('energy', 0)))
        else:
            logging.warning(f"No metadata found for {track['filename']}. Track will be skipped.")
            unmatched_tracks.append(track)
            tracks.remove(track)

    if unmatched_tracks:
        logging.info(f"{len(unmatched_tracks)} tracks were skipped due to missing metadata.")

    return tracks


def extract_artist_and_title(filename):
    parts = filename.split(' - ')
    if len(parts) == 2:
        artist, title = parts
    else:
        artist, title = '', parts[0]
    return artist.strip(), title.strip()

def extract_librosa_features(y, sr):
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).flatten()
        mfcc = librosa.feature.mfcc(y=y, sr=sr).flatten()
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).flatten()
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).flatten()
        zcr = librosa.feature.zero_crossing_rate(y).flatten()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).flatten()
        rms = librosa.feature.rms(y=y).flatten()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()

        harmonicity = librosa.effects.harmonic(y).flatten()
        onset_env = librosa.onset.onset_strength(y=y, sr=sr).flatten()

        combined_features = np.hstack([
            chroma, mfcc, spectral_contrast, tonnetz,
            zcr, spectral_rolloff, rms, spectral_centroid,
            harmonicity, onset_env
        ])

        return combined_features, tempo, beat_frames
    
    except Exception as e:
        logging.error(f"Failed to extract features: {e}")
        raise

def analyze_tracks_with_librosa(tracks, target_length=1024):
    scaler = StandardScaler()
    all_features = []

    for track in tqdm(tracks, desc="Analyzing Tracks with Librosa"):
        try:
            y, sr = librosa.load(track['filename'], sr=None)
            combined_features, tempo, beat_frames = extract_librosa_features(y, sr)

            track_length = librosa.get_duration(y=y, sr=sr)
            dynamic_range = np.max(y) - np.min(y)

            track['features'] = pad_or_truncate(combined_features, target_length)
            track['length'] = track_length
            track['dynamic_range'] = dynamic_range
            track['tempo'] = tempo
            track['beat_frames'] = beat_frames
            all_features.append(track['features'])

        except Exception as e:
            logging.error(f"Error processing {track['filename']}: {e}")
            continue

    all_features = scaler.fit_transform(all_features)
    for idx, track in enumerate(tracks):
        track['features'] = all_features[idx]

    return tracks

def pad_or_truncate(features, target_length):
    if len(features) > target_length:
        return features[:target_length]
    elif len(features) < target_length:
        return np.pad(features, (0, target_length - len(features)), 'constant')
    return features

def analyze_tracks(directory, csv_file_path, progress_callback=None):
    tracks = []
    audio_files = [f for f in os.listdir(directory) if f.endswith('.mp3') or f.endswith('.wav')]
    total_files = len(audio_files)

    for idx, filename in enumerate(audio_files):
        file_path = os.path.join(directory, filename)
        try:
            y, sr = librosa.load(file_path, sr=None)
            combined_features, tempo, beat_frames = extract_librosa_features(y, sr)
            track_length = librosa.get_duration(y=y, sr=sr)
            dynamic_range = np.max(y) - np.min(y)

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

            if progress_callback:
                progress_callback((idx + 1) / total_files * 100)

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

    if csv_file_path:
        tracks = update_track_metadata(tracks, csv_file_path)

    return tracks

def prompt_for_files():
    root = Tk()
    root.withdraw()

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
