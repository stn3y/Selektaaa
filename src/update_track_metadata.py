import csv
import os
import re
import logging

def normalize_filename(filename):
    """Normalize the filename by removing non-alphanumeric characters and converting to lowercase."""
    filename = filename.lower()
    filename = re.sub(r'[^a-z0-9]', '', filename)
    return filename

def update_track_metadata(tracks, csv_file_path):
    """Update the track metadata from the Mixed In Key CSV file."""
    try:
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            mixed_in_key_data = {
                normalize_filename(os.path.splitext(row['Filename'])[0]): row for row in reader
            }
    except KeyError:
        print("Error: CSV file is missing the 'Filename' column.")
        return tracks

    for track in tracks:
        normalized_name = normalize_filename(os.path.splitext(track['filename'])[0])
        if normalized_name in mixed_in_key_data:
            metadata = mixed_in_key_data[normalized_name]
            track.update({
                'tempo': float(metadata.get('Tempo', 0)),
                'key': metadata.get('Key'),
                'energy': float(metadata.get('Energy', 0)),
                'length': float(metadata.get('Length', 0)),
                'dynamic_range': float(metadata.get('Dynamic Range', 0)),
                'harmonicity': float(metadata.get('Harmonicity', 0)),
                'onset_strength': float(metadata.get('Onset Strength', 0)),
                'rhythm': float(metadata.get('Rhythm', 0))
            })
        else:
            logging.warning(f"No metadata found for {track['filename']}")
    
    return tracks