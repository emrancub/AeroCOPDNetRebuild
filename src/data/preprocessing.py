from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import librosa

def load_audio_segment(path: str, sr: int, start_sec: float, end_sec: float, max_len_sec: float) -> Tuple[np.ndarray, int]:
    # librosa loads mono by default
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    n = len(y)
    if end_sec is None or end_sec < 0:
        seg = y
    else:
        s = int(max(0.0, start_sec) * sr)
        e = int(min(end_sec, n / sr) * sr)
        seg = y[s:e] if e > s else np.zeros(1, dtype=np.float32)

    # pad/trim to max_len_sec (helps batching)
    max_len = int(max_len_sec * sr)
    if len(seg) < max_len:
        seg = np.pad(seg, (0, max_len - len(seg)))
    elif len(seg) > max_len:
        seg = seg[:max_len]
    return seg.astype(np.float32), sr

def highpass_filter(y: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    # simple first-order high-pass via librosa effects
    try:
        import scipy.signal as signal
        b, a = signal.butter(2, cutoff_hz / (0.5 * sr), btype="highpass")
        return signal.filtfilt(b, a, y).astype(np.float32)
    except Exception:
        return y
