import numpy as np
import soundfile as sf
import librosa
import logging

class PlaylistEnvironment:
    def __init__(self, tracks, user_input=None, reward_weights=None, crossfade_duration=5):
        self.tracks = tracks
        self.user_input = user_input or {}
        self.reward_weights = reward_weights or {
            'similarity': 1.0,
            'tempo': 0.5,
            'key': 0.5,
            'energy': 0.5,
            'length': 0.3,
            'dynamic_range': 0.3,
            'harmonicity': 0.4,
            'onset_strength': 0.4,
            'rhythm': 0.4
        }
        self.crossfade_duration = crossfade_duration
        self.current_playlist = []
        self.available_tracks = list(range(len(tracks)))
        self.done = False
        self.feature_dim = len(tracks[0]['features']) if tracks else 0
        logging.info(f"Initialized PlaylistEnvironment with {len(self.tracks)} tracks.")

        # Set dynamic context based on user input or track analysis
        self.context_mapping = self.generate_dynamic_context_mapping()

    def generate_dynamic_context_mapping(self):
        if self.user_input:
            # User-driven context mapping
            return {
                "tempo": self.user_input.get("tempo", 120),
                "energy": self.user_input.get("energy", 5)
            }
        else:
            # Adaptive context mapping based on track analysis
            average_tempo = np.mean([track.get('tempo', 120) for track in self.tracks])
            average_energy = np.mean([track.get('energy', 5) for track in self.tracks])
            
            return {
                "tempo": average_tempo,
                "energy": average_energy
            }

    def reset(self):
        self.current_playlist = []
        self.done = False
        self.available_tracks = list(range(len(self.tracks)))
        return self.get_state()

    def get_state(self):
        if not self.current_playlist:
            return np.zeros(self.feature_dim)
        return np.mean([track['features'] for track in self.current_playlist], axis=0)

    def step(self, action):
        if action not in self.available_tracks:
            raise ValueError(f"Action {action} is not available in the current track list.")

        track = self.tracks[action]
        self.current_playlist.append(track)
        self.available_tracks.remove(action)

        reward = 0.0
        if len(self.current_playlist) > 1:
            reward = self.calculate_reward(self.current_playlist[-2], track)
        if len(self.current_playlist) >= len(self.tracks) or not self.available_tracks:
            self.done = True

        return self.get_state(), reward, self.done, self.available_tracks

    def calculate_reward(self, track1, track2):
        # Ensure feature vectors are the same length
        min_length = min(len(track1['features']), len(track2['features']))
        track1_features = track1['features'][:min_length]
        track2_features = track2['features'][:min_length]

        # Calculate similarity reward
        similarity = np.dot(track1_features, track2_features) / (
            np.linalg.norm(track1_features) * np.linalg.norm(track2_features))
        reward = self.reward_weights.get('similarity', 1.0) * similarity

        # Tempo consistency
        if 'tempo' in track1 and 'tempo' in track2:
            reward += self.reward_weights.get('tempo', 0.2) * (abs(track1['tempo'] - track2['tempo']) < 5)

        # Key consistency
        if 'key' in track1 and 'key' in track2:
            reward += self.reward_weights.get('key', 0.2) * (track1['key'] == track2['key'])

        # Energy consistency
        if 'energy' in track1 and 'energy' in track2:
            reward += self.reward_weights.get('energy', 0.2) * (abs(track1['energy'] - track2['energy']) < 2)

        # Length consideration: Favor varying track lengths to create dynamic flow
        if 'length' in track1 and 'length' in track2:
            length_diff = abs(track1['length'] - track2['length'])
            if length_diff < 30:
                reward += self.reward_weights.get('length', 0.2) * (1 - length_diff / 30)  # Closer lengths get less reward
            else:
                reward += self.reward_weights.get('length', 0.2) * (length_diff / max(track1['length'], track2['length']))

        # Dynamic range consistency: Favor smooth transitions in energy
        if 'dynamic_range' in track1 and 'dynamic_range' in track2:
            dynamic_range_diff = abs(track1['dynamic_range'] - track2['dynamic_range'])
            reward += self.reward_weights.get('dynamic_range', 0.2) * (1 - dynamic_range_diff / max(track1['dynamic_range'], track2['dynamic_range']))

        # Harmonicity consistency
        if 'harmonicity' in track1 and 'harmonicity' in track2:
            harmonicity_diff = abs(track1['harmonicity'] - track2['harmonicity'])
            reward += self.reward_weights.get('harmonicity', 0.2) * (1 - harmonicity_diff / max(track1['harmonicity'], track2['harmonicity']))

        # Onset detection consistency (onset strength)
        if 'onset_strength' in track1 and 'onset_strength' in track2:
            onset_diff = abs(track1['onset_strength'] - track2['onset_strength'])
            reward += self.reward_weights.get('onset_strength', 0.2) * (1 - onset_diff / max(track1['onset_strength'], track2['onset_strength']))

        # Rhythmic pattern consistency
        if 'rhythm' in track1 and 'rhythm' in track2:
            rhythm_diff = abs(track1['rhythm'] - track2['rhythm'])
            reward += self.reward_weights.get('rhythm', 0.2) * (1 - rhythm_diff / max(track1['rhythm'], track2['rhythm']))

        return reward    
    def generate_context_aware_playlist(self, user_context):
            # Generate dynamic context if not provided
            context_attributes = self.context_mapping or context_mapping.get(user_context, {})
            filtered_tracks = []

            default_tempo = 120  # Example default tempo, adjust as needed
            default_energy = 5   # Example default energy, adjust as needed

            for track in self.tracks:
                track_tempo = track.get('tempo', default_tempo)
                track_energy = track.get('energy', default_energy)

                if "tempo" in context_attributes and track_tempo == context_attributes['tempo']:
                    filtered_tracks.append(track)
                elif "energy" in context_attributes and track_energy == context_attributes['energy']:
                    filtered_tracks.append(track)

            if not filtered_tracks:
                filtered_tracks = self.tracks

            return filtered_tracks

    def sequence_tracks_by_flow(self, filtered_tracks, user_context):
        default_tempo = 120  # Default tempo
        default_energy = 5   # Default energy
        track_history = []

        for track in filtered_tracks:
            track['tempo'] = track.get('tempo', default_tempo)
            track['energy'] = track.get('energy', default_energy)

        # Sort tracks by multiple criteria including tempo, energy, length, dynamic range, harmonicity, onset strength, and rhythm
        sorted_tracks = sorted(filtered_tracks, key=lambda t: (
            t['tempo'], t['energy'], t['length'], t['dynamic_range'], 
            t.get('harmonicity', 0), t.get('onset_strength', 0), t.get('rhythm', 0)
        ))

        sequenced_tracks = []
        for track in sorted_tracks:
            if not track_history or self.is_track_unique(track, track_history):
                sequenced_tracks.append(track)
                track_history.append(track)
                if len(track_history) > 5:  # Maintain a history of the last 5 tracks
                    track_history.pop(0)

        return sequenced_tracks

    def is_track_unique(self, track, track_history):
        for previous_track in track_history:
            if abs(track['tempo'] - previous_track['tempo']) < 5 and \
               abs(track['energy'] - previous_track['energy']) < 2 and \
               abs(track['length'] - previous_track['length']) < 30 and \
               abs(track['dynamic_range'] - previous_track['dynamic_range']) < 0.1 and \
               abs(track.get('harmonicity', 0) - previous_track.get('harmonicity', 0)) < 0.1 and \
               abs(track.get('onset_strength', 0) - previous_track.get('onset_strength', 0)) < 0.1 and \
               abs(track.get('rhythm', 0) - previous_track.get('rhythm', 0)) < 0.1:
                return False
        return True

    def apply_crossfade(self, track1, track2):
        try:
            y1, sr1 = librosa.load(track1['filename'], sr=None)
        except Exception as e:
            logging.error(f"Failed to load {track1['filename']}: {e}")
            return None

        try:
            y2, sr2 = librosa.load(track2['filename'], sr=None)
        except Exception as e:
            logging.error(f"Failed to load {track2['filename']}: {e}")
            return None

        if sr1 != sr2:
            logging.warning(f"Sample rates differ between tracks: {sr1} vs {sr2}. Resampling...")
            y2 = librosa.resample(y2, sr2, sr1)
            sr2 = sr1

        # Proceed with the crossfade if both tracks are successfully loaded
        crossfade_duration = min(len(y1), len(y2)) // 4  # Example: 25% of the shorter track's duration
        fade_out = np.linspace(1, 0, crossfade_duration)
        fade_in = np.linspace(0, 1, crossfade_duration)

        y1[-crossfade_duration:] *= fade_out
        y2[:crossfade_duration] *= fade_in

        output_audio = np.concatenate([y1[:-crossfade_duration], y1[-crossfade_duration:] + y2[:crossfade_duration], y2[crossfade_duration:]])
        return output_audio, sr1

    def normalize_volume(self, y, target_dBFS=-20.0):
        rms = np.sqrt(np.mean(y**2))
        current_dBFS = 20 * np.log10(rms)
        scaling_factor = 10**((target_dBFS - current_dBFS) / 20)
        return y * scaling_factor

    def calculate_dynamic_crossfade(self, track1, track2):
        # Adjust crossfade duration based on tempo, energy, and dynamic range differences
        tempo_diff = abs(track1.get('tempo', 120) - track2.get('tempo', 120))
        energy_diff = abs(track1.get('energy', 5) - track2.get('energy', 5))
        dynamic_range_diff = abs(track1.get('dynamic_range', 0) - track2.get('dynamic_range', 0))

        # Increase crossfade duration if tempo, energy, or dynamic range differences are significant
        base_crossfade = self.crossfade_duration
        adjustment_factor = 1 + (tempo_diff + energy_diff + dynamic_range_diff) / 10
        return base_crossfade * adjustment_factor

    def save_playlist_with_crossfade(self, playlist, output_path):
        if len(playlist) < 2:
            raise ValueError("Playlist must contain at least two tracks to apply crossfade.")

        output_audio = self.apply_crossfade(playlist[0], playlist[1])

        for i in range(2, len(playlist)):
            output_audio = np.concatenate([output_audio, self.apply_crossfade(playlist[i-1], playlist[i])])

        sf.write(output_path, output_audio, 44100)

    def collect_user_feedback(self, playlist):
        # Placeholder for user feedback collection
        # In a real application, this could be a UI input, a survey, or a feedback form
        logging.info("Collecting user feedback...")
        feedback = {}  # Example: {"overall_rating": 4.5, "transition_ratings": [4, 5, 3, 4]}
        return feedback

    def apply_user_feedback(self, feedback):
        logging.info("Applying user feedback to adjust rewards...")
        # Adjust reward weights based on feedback
        self.reward_weights['similarity'] *= feedback.get('overall_rating', 1)
        # Adjust other rewards based on transition ratings, etc.

# Example context mapping (to be included or imported elsewhere in your application)
context_mapping = {
    "warm_up_set": {"tempo": 100, "energy": 3},
    "peak_hour_set": {"tempo": 128, "energy": 8},
    "cool_down_set": {"tempo": 90, "energy": 2},
    "after_party": {"tempo": 120, "energy": 5},
    "chill_out": {"tempo": 85, "energy": 1}
}