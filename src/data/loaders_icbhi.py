from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .preprocessing import load_audio_segment, highpass_filter
from .features import FeatureConfig, waveform_to_logmel, normalize_mel

class LungSoundDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        indices: np.ndarray,
        feature_cfg: FeatureConfig,
        normalize_mode: str,
        train_mean: Optional[np.ndarray] = None,
        train_std: Optional[np.ndarray] = None,
    ):
        self.df = df.loc[indices].reset_index(drop=True)
        self.feature_cfg = feature_cfg
        self.normalize_mode = normalize_mode
        self.train_mean = train_mean
        self.train_std = train_std

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        r = self.df.iloc[i]
        path = str(r["filepath"])
        start = float(r.get("start_sec", 0.0))
        end = float(r.get("end_sec", -1.0))
        y, sr = load_audio_segment(path, self.feature_cfg.sample_rate, start, end, self.feature_cfg.max_len_sec)
        if self.feature_cfg.highpass_hz is not None:
            y = highpass_filter(y, sr, float(self.feature_cfg.highpass_hz))
        mel = waveform_to_logmel(y, self.feature_cfg)
        mel = normalize_mel(mel, self.normalize_mode, self.train_mean, self.train_std)
        x = torch.from_numpy(mel).unsqueeze(0)  # [1,F,T]
        ylab = torch.tensor(float(r["label"]), dtype=torch.float32)
        return {
            "x": x,
            "y": ylab,
            "sample_id": str(r["sample_id"]),
            "subject_id": str(r["subject_id"]),
            "dataset": str(r["dataset"]),
            "filepath": str(r["filepath"]),
        }
