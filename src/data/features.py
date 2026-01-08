from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import librosa

@dataclass
class FeatureConfig:
    sample_rate: int = 4000
    highpass_hz: Optional[float] = 50.0
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 64
    fmin: int = 50
    fmax: int = 2000
    log_mel: bool = True
    normalize: str = "train_stats"  # train_stats|per_sample|none
    max_len_sec: float = 6.0

def waveform_to_logmel(y: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    if cfg.log_mel:
        S = librosa.power_to_db(S, ref=np.max)
    # output shape: [n_mels, time]
    return S.astype(np.float32)

def compute_train_stats(mels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # mels shape: [N, F, T]
    mean = mels.mean(axis=(0,2), keepdims=True)
    std  = mels.std(axis=(0,2), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

def normalize_mel(mel: np.ndarray, mode: str, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    """Normalize log-mel spectrogram.

    mel: (n_mels, n_frames)
    mean/std:
      - if mode == 'global': mean/std are scalars or shape (1,)
      - if mode == 'per_mel': mean/std can be shape (n_mels,) or (n_mels,1)
    """
    if mode is None or mode.lower() in ["none", "no", "off"]:
        return mel.astype(np.float32)

    if mean is None or std is None:
        return mel.astype(np.float32)

    mode = mode.lower()

    # Backward/compat aliases from configs
    if mode in ["train_stats", "train", "trainstat", "train_stats_per_mel"]:
        mode = "per_mel"
    if mode in ["train_stats_global", "train_global"]:
        mode = "global"

    # Convert to arrays
    mean = np.asarray(mean)
    std = np.asarray(std)

    if mode == "global":
        m = float(mean.reshape(-1)[0])
        s = float(std.reshape(-1)[0]) if float(std.reshape(-1)[0]) > 0 else 1.0
        return ((mel - m) / s).astype(np.float32)

    if mode in ["per_mel", "per_mel_bin", "mel"]:
        # mean/std should broadcast across time axis
        if mean.ndim == 1:
            mean = mean[:, None]  # (n_mels,1)
        if std.ndim == 1:
            std = std[:, None]
        std = np.where(std == 0, 1.0, std)
        return ((mel - mean) / std).astype(np.float32)

    raise ValueError(f"Unknown normalize mode: {mode}")

