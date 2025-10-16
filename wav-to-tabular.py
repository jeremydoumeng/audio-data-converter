import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


class AudioTabularFeatureExtractor:
    """Extracteur complet de features audio pour modèles tabulaires"""

    def __init__(self, sample_rate=16000, n_mels=64, n_fft=1024,
                 hop_length=256, n_mfcc=20):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            power=2.0
        )

        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )

        self.spectral_centroid = T.SpectralCentroid(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )

    def load_audio(self, audio_path, duration=None):
        """Charge un fichier audio"""
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if duration is not None:
            num_samples = int(duration * self.sample_rate)
            if waveform.shape[1] > num_samples:
                waveform = waveform[:, :num_samples]
            else:
                padding = num_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform

    def compute_statistics(self, data, prefix=""):
        """Calcule des statistiques descriptives sur un tensor"""
        data_np = data.cpu().numpy().flatten()

        features = {
            f"{prefix}_mean": np.mean(data_np),
            f"{prefix}_std": np.std(data_np),
            f"{prefix}_median": np.median(data_np),
            f"{prefix}_min": np.min(data_np),
            f"{prefix}_max": np.max(data_np),
            f"{prefix}_q25": np.percentile(data_np, 25),
            f"{prefix}_q75": np.percentile(data_np, 75),
            f"{prefix}_iqr": np.percentile(data_np, 75) - np.percentile(data_np, 25),
            f"{prefix}_skewness": stats.skew(data_np),
            f"{prefix}_kurtosis": stats.kurtosis(data_np)
        }

        return features

    def compute_temporal_features(self, data, prefix="", n_coefs=5):
        """Calcule des features temporelles sur chaque coefficient"""
        features = {}
        n_coefs = min(n_coefs, data.shape[0])

        for i in range(n_coefs):
            coef_data = data[i, :].cpu().numpy()

            x = np.arange(len(coef_data))
            if len(coef_data) > 1:
                slope, intercept = np.polyfit(x, coef_data, 1)
                features[f"{prefix}_coef{i}_slope"] = slope
                features[f"{prefix}_coef{i}_intercept"] = intercept

            if len(coef_data) > 10:
                window_size = min(10, len(coef_data) // 4)
                local_stds = [np.std(coef_data[j:j + window_size])
                              for j in range(0, len(coef_data) - window_size, window_size)]
                features[f"{prefix}_coef{i}_local_std_mean"] = np.mean(local_stds)
                features[f"{prefix}_coef{i}_local_std_std"] = np.std(local_stds)

        return features

    def extract_waveform_features(self, waveform):
        """Features basiques du waveform"""
        waveform_np = waveform.squeeze().cpu().numpy()
        features = {}

        features['energy_total'] = np.sum(waveform_np ** 2)
        features['energy_mean'] = np.mean(waveform_np ** 2)

        zero_crossings = np.sum(np.diff(np.sign(waveform_np)) != 0)
        features['zcr'] = zero_crossings / len(waveform_np)
        features['rms'] = np.sqrt(np.mean(waveform_np ** 2))

        features.update(self.compute_statistics(waveform, "waveform"))

        return features

    def compute_deltas(self, data, win_length=5):
        """Calcule les deltas"""
        n = (win_length - 1) // 2
        denom = n * (n + 1) * (2 * n + 1) / 3

        data_padded = torch.nn.functional.pad(
            data.unsqueeze(0).unsqueeze(0),
            (n, n, 0, 0),
            mode='replicate'
        ).squeeze(0).squeeze(0)

        deltas = torch.zeros_like(data)
        for t in range(data.shape[1]):
            acc = 0
            for k in range(-n, n + 1):
                acc += k * data_padded[:, t + n + k]
            deltas[:, t] = acc / denom

        return deltas

    def extract_all_features(self, audio_path, duration=None):
        """Extrait toutes les features tabulaires d'un fichier audio"""
        waveform = self.load_audio(audio_path, duration)
        features = {}

        # Waveform
        features.update(self.extract_waveform_features(waveform))

        # MFCC
        mfcc = self.mfcc_transform(waveform).squeeze(0)
        features.update(self.compute_statistics(mfcc, "mfcc"))

        for i in range(self.n_mfcc):
            features.update(self.compute_statistics(mfcc[i, :], f"mfcc_c{i}"))

        features.update(self.compute_temporal_features(mfcc, "mfcc", n_coefs=10))

        # Deltas
        delta = self.compute_deltas(mfcc, win_length=5)
        features.update(self.compute_statistics(delta, "delta"))

        for i in range(min(10, self.n_mfcc)):
            features.update(self.compute_statistics(delta[i, :], f"delta_c{i}"))

        features.update(self.compute_temporal_features(delta, "delta", n_coefs=5))

        # Delta-Deltas
        delta_delta = self.compute_deltas(delta, win_length=5)
        features.update(self.compute_statistics(delta_delta, "delta_delta"))

        for i in range(min(5, self.n_mfcc)):
            features.update(self.compute_statistics(delta_delta[i, :], f"dd_c{i}"))

        features.update(self.compute_temporal_features(delta_delta, "dd", n_coefs=3))

        # Mel Spectrogram
        mel_spec = self.mel_spectrogram(waveform).squeeze(0)
        features.update(self.compute_statistics(mel_spec, "mel_spec"))

        # Spectral Centroid
        centroid = self.spectral_centroid(waveform).squeeze(0)
        features.update(self.compute_statistics(centroid, "spectral_centroid"))

        return features


def create_tabular_dataset(
        input_dir,
        output_csv="animal_sounds_features.csv",
        duration=None,
        sample_rate=16000,
        n_mfcc=20,
        file_extensions=['.wav', '.mp3', '.flac']
):
    """Crée un dataset tabulaire CSV à partir d'un dossier d'audios classifiés"""

    input_path = Path(input_dir)
    extractor = AudioTabularFeatureExtractor(sample_rate=sample_rate, n_mfcc=n_mfcc)

    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if len(class_dirs) == 0:
        print(f"Aucun sous-dossier trouvé dans {input_dir}")
        return None

    all_features = []
    total_files = sum(len(list(class_dir.glob(f"*{ext}")))
                      for class_dir in class_dirs
                      for ext in file_extensions)

    print(f"Extraction de {total_files} fichiers...")

    for class_dir in class_dirs:
        class_name = class_dir.name
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(list(class_dir.glob(f"*{ext}")))

        for audio_file in audio_files:
            try:
                features = extractor.extract_all_features(str(audio_file), duration)
                features['class'] = class_name
                all_features.append(features)
            except Exception as e:
                print(f"Erreur sur {audio_file.name}: {e}")
                continue

    df = pd.DataFrame(all_features)
    cols = ['class'] + [c for c in df.columns if c != 'class']
    df = df[cols]

    df.to_csv(output_csv, index=False)

    print(f"\nDataset créé: {len(df)} exemples, {len(df.columns) - 1} features")
    print(f"Sauvegardé dans: {output_csv}")
    print(f"\nDistribution des classes:")
    print(df['class'].value_counts())

    return df


if __name__ == "__main__":
    df = create_tabular_dataset(
        input_dir="/Users/jeremy/PycharmProjects/animalaudio/animal noise dataset /Animal-Soundprepros",
        output_csv="./animal_sounds_complete.csv",
        duration=None,
        sample_rate=16000,
        n_mfcc=20,
        file_extensions=['.wav', '.mp3', '.flac']
    )