import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np


class SpectrogramBatchGenerator:
    """Générateur batch de spectrogrammes, MFCC, Deltas et Deltas-Deltas"""

    def __init__(self, sample_rate=16000, n_mels=64, n_fft=1024,
                 hop_length=256, f_min=20, f_max=None, normalize=True,
                 to_db=True, top_db=80, n_mfcc=20):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize
        self.to_db = to_db
        self.n_mfcc = n_mfcc

        if f_max is None:
            f_max = sample_rate / 2

        # Transformation mel-spectrogramme
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode='reflect',
            norm='slaney',
            mel_scale='htk'
        )

        if to_db:
            self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=top_db)
        else:
            self.amplitude_to_db = None

        # Transformation MFCC
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels,
                'f_min': f_min,
                'f_max': f_max,
                'window_fn': torch.hann_window,
                'power': 2.0,
                'normalized': False,
                'center': True,
                'pad_mode': 'reflect',
                'norm': 'slaney',
                'mel_scale': 'htk'
            }
        )

    def load_audio(self, audio_path, duration=None, offset=0.0):
        """Charge et prétraite un fichier audio"""
        waveform, sr = torchaudio.load(audio_path)

        # Resample si nécessaire
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Convertir en mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Extraire la portion voulue
        if duration is not None:
            start_sample = int(offset * self.sample_rate)
            num_samples = int(duration * self.sample_rate)
            end_sample = start_sample + num_samples

            if end_sample <= waveform.shape[1]:
                waveform = waveform[:, start_sample:end_sample]
            else:
                waveform = waveform[:, start_sample:]
                padding = num_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Normalisation amplitude
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        return waveform

    def compute_deltas(self, specgram, win_length=5, mode='replicate'):
        """
        Calcule les deltas (dérivées premières) d'un spectrogramme ou MFCC

        Args:
            specgram: Tensor de forme (n_features, time)
            win_length: Longueur de la fenêtre pour le calcul des deltas
            mode: Mode de padding ('replicate', 'reflect', 'constant')

        Returns:
            Tensor des deltas de même forme que l'entrée
        """
        # Créer les coefficients pour le calcul des deltas
        n = (win_length - 1) // 2
        denom = n * (n + 1) * (2 * n + 1) / 3

        # Padding
        specgram_padded = torch.nn.functional.pad(
            specgram.unsqueeze(0).unsqueeze(0),
            (n, n, 0, 0),
            mode=mode
        ).squeeze(0).squeeze(0)

        # Calcul des deltas
        deltas = torch.zeros_like(specgram)
        for t in range(specgram.shape[1]):
            acc = 0
            for k in range(-n, n + 1):
                acc += k * specgram_padded[:, t + n + k]
            deltas[:, t] = acc / denom

        return deltas

    def generate_spectrogram(self, audio_input, duration=None):
        """Génère un mel-spectrogramme"""
        if isinstance(audio_input, (str, Path)):
            waveform = self.load_audio(audio_input, duration=duration)
        else:
            waveform = audio_input
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        mel_spec = self.mel_spectrogram(waveform)

        if self.to_db and self.amplitude_to_db is not None:
            mel_spec = self.amplitude_to_db(mel_spec)

        mel_spec = mel_spec.squeeze(0)

        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        return mel_spec

    def generate_mfcc(self, audio_input, duration=None):
        """Génère les coefficients MFCC"""
        if isinstance(audio_input, (str, Path)):
            waveform = self.load_audio(audio_input, duration=duration)
        else:
            waveform = audio_input
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0)

        if self.normalize:
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)

        return mfcc

    def generate_mfcc_with_deltas(self, audio_input, duration=None, win_length=5):
        """
        Génère MFCC avec Deltas et Deltas-Deltas

        Returns:
            dict avec clés 'mfcc', 'delta', 'delta_delta', 'combined'
        """
        # Générer MFCC
        mfcc = self.generate_mfcc(audio_input, duration)

        # Calculer Deltas
        delta = self.compute_deltas(mfcc, win_length=win_length)

        # Calculer Deltas-Deltas (deltas des deltas)
        delta_delta = self.compute_deltas(delta, win_length=win_length)

        # Combiner tout (empilage vertical)
        combined = torch.cat([mfcc, delta, delta_delta], dim=0)

        return {
            'mfcc': mfcc,
            'delta': delta,
            'delta_delta': delta_delta,
            'combined': combined
        }

    def save_as_image(self, data, save_path, title, data_type="spectrogram"):
        """Sauvegarde un spectrogramme ou MFCC comme image PNG"""
        plt.figure(figsize=(10, 4))

        data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data

        cmap = 'viridis' if data_type == "spectrogram" else 'coolwarm'

        img = plt.imshow(
            data_np,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            interpolation='nearest'
        )

        if data_type == "spectrogram":
            plt.colorbar(img, format='%+2.0f dB' if self.to_db else '%+2.2f')
            plt.ylabel('Fréquence Mel')
        else:
            plt.colorbar(img, format='%+2.2f')
            plt.ylabel('Coefficient MFCC')

        plt.title(title, fontsize=10)
        plt.xlabel('Temps (frames)')
        plt.tight_layout()

        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    def save_as_tensor(self, data, save_path):
        """Sauvegarde comme tensor PyTorch (.pt)"""
        torch.save(data, save_path)


def process_dataset(
        input_dir,
        output_dir,
        duration=None,
        save_format="image",
        sample_rate=16000,
        n_mels=64,
        n_mfcc=20,
        file_extensions=['.wav'],
        include_deltas=True,
        delta_on_spectrogram=False
):
    """
    Traite tout le dataset et génère spectrogrammes, MFCC, Deltas et Deltas-Deltas

    Structure sortie (avec include_deltas=True):
        output_dir/
            dog_spectro/
            dog_mfcc/
            dog_delta/
            dog_delta_delta/
            dog_mfcc_combined/  # MFCC + Delta + Delta-Delta empilés
            cat_spectro/
            cat_mfcc/
            cat_delta/
            cat_delta_delta/
            cat_mfcc_combined/

    Args:
        input_dir: Dossier contenant les sous-dossiers par classe
        output_dir: Dossier de sortie
        duration: Durée fixe en secondes (None = durée complète)
        save_format: "image" (PNG), "tensor" (.pt), ou "both"
        sample_rate: Taux d'échantillonnage
        n_mels: Nombre de bandes mel
        n_mfcc: Nombre de coefficients MFCC
        file_extensions: Extensions de fichiers audio à traiter
        include_deltas: Si True, génère aussi les deltas et deltas-deltas
        delta_on_spectrogram: Si True, applique aussi les deltas sur le spectrogramme
    """
    print("=" * 80)
    print("GÉNÉRATION BATCH DE SPECTROGRAMMES, MFCC ET DELTAS")
    print("=" * 80)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialiser le générateur
    generator = SpectrogramBatchGenerator(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_mfcc=n_mfcc
    )

    # Trouver toutes les classes
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if len(class_dirs) == 0:
        print(f"Aucun sous-dossier trouvé dans {input_dir}")
        return

    print(f"\n🏷️  Classes trouvées: {[d.name for d in class_dirs]}")

    total_files = 0
    total_processed = 0
    total_errors = 0

    # Parcourir chaque classe
    for class_dir in class_dirs:
        class_name = class_dir.name

        # Créer les dossiers de sortie
        spectro_dir = output_path / f"{class_name}_spectro"
        mfcc_dir = output_path / f"{class_name}_mfcc"
        spectro_dir.mkdir(parents=True, exist_ok=True)
        mfcc_dir.mkdir(parents=True, exist_ok=True)

        if include_deltas:
            delta_dir = output_path / f"{class_name}_delta"
            delta_delta_dir = output_path / f"{class_name}_delta_delta"
            combined_dir = output_path / f"{class_name}_mfcc_combined"
            delta_dir.mkdir(parents=True, exist_ok=True)
            delta_delta_dir.mkdir(parents=True, exist_ok=True)
            combined_dir.mkdir(parents=True, exist_ok=True)

        if delta_on_spectrogram:
            spectro_delta_dir = output_path / f"{class_name}_spectro_delta"
            spectro_delta_delta_dir = output_path / f"{class_name}_spectro_delta_delta"
            spectro_combined_dir = output_path / f"{class_name}_spectro_combined"
            spectro_delta_dir.mkdir(parents=True, exist_ok=True)
            spectro_delta_delta_dir.mkdir(parents=True, exist_ok=True)
            spectro_combined_dir.mkdir(parents=True, exist_ok=True)

        # Trouver tous les fichiers audio
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(list(class_dir.glob(f"*{ext}")))

        total_files += len(audio_files)


        # Traiter chaque fichier audio
        for audio_file in audio_files:
            try:
                base_name = audio_file.stem

                # Générer spectrogramme
                spectrogram = generator.generate_spectrogram(str(audio_file), duration=duration)

                # Générer deltas du spectrogramme si demandé
                if delta_on_spectrogram:
                    spectro_delta = generator.compute_deltas(spectrogram, win_length=5)
                    spectro_delta_delta = generator.compute_deltas(spectro_delta, win_length=5)
                    spectro_combined = torch.cat([spectrogram, spectro_delta, spectro_delta_delta], dim=0)

                # Générer MFCC avec deltas
                if include_deltas:
                    mfcc_data = generator.generate_mfcc_with_deltas(str(audio_file), duration=duration)
                    mfcc = mfcc_data['mfcc']
                    delta = mfcc_data['delta']
                    delta_delta = mfcc_data['delta_delta']
                    combined = mfcc_data['combined']
                else:
                    mfcc = generator.generate_mfcc(str(audio_file), duration=duration)

                # Sauvegarder spectrogramme
                if save_format in ["image", "both"]:
                    spec_path = spectro_dir / f"{base_name}.png"
                    generator.save_as_image(
                        spectrogram,
                        spec_path,
                        f"{class_name} - {base_name}",
                        "spectrogram"
                    )

                if save_format in ["tensor", "both"]:
                    spec_path = spectro_dir / f"{base_name}.pt"
                    generator.save_as_tensor(spectrogram, spec_path)

                # Sauvegarder MFCC
                if save_format in ["image", "both"]:
                    mfcc_path = mfcc_dir / f"{base_name}.png"
                    generator.save_as_image(
                        mfcc,
                        mfcc_path,
                        f"{class_name} MFCC - {base_name}",
                        "mfcc"
                    )

                if save_format in ["tensor", "both"]:
                    mfcc_path = mfcc_dir / f"{base_name}.pt"
                    generator.save_as_tensor(mfcc, mfcc_path)

                # Sauvegarder Deltas et Deltas-Deltas
                if include_deltas:
                    if save_format in ["image", "both"]:
                        delta_path = delta_dir / f"{base_name}.png"
                        generator.save_as_image(
                            delta,
                            delta_path,
                            f"{class_name} Delta - {base_name}",
                            "mfcc"
                        )

                        delta_delta_path = delta_delta_dir / f"{base_name}.png"
                        generator.save_as_image(
                            delta_delta,
                            delta_delta_path,
                            f"{class_name} Delta-Delta - {base_name}",
                            "mfcc"
                        )

                        combined_path = combined_dir / f"{base_name}.png"
                        generator.save_as_image(
                            combined,
                            combined_path,
                            f"{class_name} MFCC+Deltas - {base_name}",
                            "mfcc"
                        )

                    if save_format in ["tensor", "both"]:
                        delta_path = delta_dir / f"{base_name}.pt"
                        generator.save_as_tensor(delta, delta_path)

                        delta_delta_path = delta_delta_dir / f"{base_name}.pt"
                        generator.save_as_tensor(delta_delta, delta_delta_path)

                        combined_path = combined_dir / f"{base_name}.pt"
                        generator.save_as_tensor(combined, combined_path)

                total_processed += 1

            except Exception as e:
                total_errors += 1
                continue


    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"  • {class_name}_spectro/")
        print(f"  • {class_name}_mfcc/")
        if include_deltas:
            print(f"  • {class_name}_delta/")
            print(f"  • {class_name}_delta_delta/")
            print(f"  • {class_name}_mfcc_combined/")


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration avec Deltas et Deltas-Deltas
    process_dataset(
        input_dir="/Users/jeremy/PycharmProjects/animalaudio/animal noise dataset /Animal-Soundprepros",
        output_dir="./processed_data",
        duration=None,
        save_format="image",  # "image", "tensor", ou "both"
        sample_rate=16000,
        n_mels=64,
        n_mfcc=20,
        include_deltas=True  # Active les deltas et deltas-deltas
    )