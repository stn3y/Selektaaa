import numpy as np
import logging

class PlaylistEnvironment:
    def __init__(self, tracks, reward_weights):
        self.tracks = [track for track in tracks if track and 'features' in track and track['features'].size > 0]
        self.reward_weights = reward_weights
        self.current_playlist = []
        self.available_tracks = list(range(len(self.tracks)))
        self.done = False

        if not self.tracks:
            logging.error("No valid tracks available for playlist generation.")
            raise ValueError("No valid tracks available for playlist generation.")

        self.feature_dim = len(self.tracks[0]['features']) if len(self.tracks) > 0 else 0
        logging.info(f"Initialized PlaylistEnvironment with {len(self.tracks)} valid tracks, feature dimension: {self.feature_dim}")

    def reset(self):
        self.current_playlist = []
        self.done = False
        self.available_tracks = list(range(len(self.tracks)))
        logging.info("Environment reset.")
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

        try:
            self.available_tracks.remove(action)
        except ValueError:
            logging.warning(f"Warning: Tried to remove action {action} that was not in available_tracks.")

        reward = 0.0
        if len(self.current_playlist) > 1:
            reward = self.calculate_reward(self.current_playlist[-2], track)
        if len(self.current_playlist) >= len(self.tracks) or not self.available_tracks:
            self.done = True

        logging.info(f"Track {track['filename']} added to playlist. Current playlist length: {len(self.current_playlist)}")

        return self.get_state(), reward, self.done, self.available_tracks

    def calculate_reward(self, track1, track2):
        total_reward = 0.0

        for reward_type, weight in self.reward_weights.items():
            if reward_type == 'similarity':
                similarity = np.dot(track1['features'], track2['features']) / (np.linalg.norm(track1['features']) * np.linalg.norm(track2['features']))
                total_reward += weight * similarity
                logging.info(f"Similarity reward: {weight * similarity}")
            elif reward_type == 'tempo':
                if 'tempo' in track1 and 'tempo' in track2:
                    tempo_diff = abs(track1['tempo'] - track2['tempo'])
                    total_reward += weight * (1.0 - (tempo_diff / 100))  # Assuming tempo is in BPM, normalize difference
                    logging.info(f"Tempo reward: {weight * (1.0 - (tempo_diff / 100))}")
                else:
                    logging.warning(f"'tempo' feature not found in tracks. Skipping tempo reward.")
            elif reward_type == 'energy':
                if 'energy' in track1 and 'energy' in track2:
                    energy_diff = abs(track1['energy'] - track2['energy'])
                    total_reward += weight * (1.0 - (energy_diff / 100))  # Normalize energy difference
                    logging.info(f"Energy reward: {weight * (1.0 - (energy_diff / 100))}")
                else:
                    logging.warning(f"'energy' feature not found in tracks. Skipping energy reward.")
            else:
                logging.warning(f"Unknown reward type: {reward_type}")

        return total_reward