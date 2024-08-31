import os
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self):
        # Load VGG16 model pre-trained on ImageNet
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])  # Remove the last layers
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, file_path):
        try:
            logging.info(f"Attempting to extract features from file: {file_path}")
            y, sr = librosa.load(file_path, sr=None)
            if y is None or len(y) == 0:
                logging.error(f"Failed to load audio data from {file_path}.")
                return None

            # Extract log-mel spectrogram as input to VGG16 model
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

            # Convert the spectrogram to a 3-channel format
            if len(log_mel_spectrogram.shape) == 2:
                log_mel_spectrogram = np.stack([log_mel_spectrogram] * 3, axis=-1)

            input_tensor = self.transform(log_mel_spectrogram).unsqueeze(0)
            with torch.no_grad():
                features = self.model(input_tensor).numpy().flatten()

            logging.info(f"Successfully extracted features from {file_path} with shape {features.shape}")
            return {
                'filename': os.path.basename(file_path),
                'features': features
            }
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None

def analyze_tracks(directory, progress_callback=None):
    extractor = AudioFeatureExtractor()
    tracks = []
    audio_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp3') or f.endswith('.wav')]

    if not audio_files:
        logging.error("No audio files found.")
        return tracks
    
    logging.info(f"Found {len(audio_files)} audio files in the directory.")

    for idx, file_path in enumerate(audio_files):
        logging.info(f"Processing file {idx + 1}/{len(audio_files)}: {file_path}")
        result = extractor.extract_features(file_path)
        if result:
            logging.info(f"Features successfully extracted for {file_path}")
            tracks.append(result)
        else:
            logging.error(f"Failed to extract features from {file_path}")

        if progress_callback:
            progress_callback((idx + 1) / len(audio_files) * 100)

    logging.info(f"Extracted features for {len(tracks)} out of {len(audio_files)} files.")
    return tracks