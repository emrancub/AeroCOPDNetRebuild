from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import re
import pandas as pd

@dataclass
class ManifestRow:
    sample_id: str
    dataset: str
    subject_id: str
    filepath: str
    label: int
    diagnosis: str
    start_sec: float
    end_sec: float

def _safe_subject_from_icbhi_filename(fname: str) -> str:
    m = re.match(r"^(\d+)_", fname)
    return m.group(1) if m else Path(fname).stem.split("_")[0]

def build_icbhi_binary(
    icbhi_root: Path,
    out_csv: Path,
    audio_subdir: str = "audio_and_txt_files",
    diag_csv_name: str = "patient_diagnosis.csv",
) -> pd.DataFrame:
    audio_dir = icbhi_root / audio_subdir
    diag_path = icbhi_root / diag_csv_name

    if not audio_dir.exists():
        raise FileNotFoundError(f"[ICBHI] audio_subdir not found: {audio_dir}")
    if not diag_path.exists():
        raise FileNotFoundError(f"[ICBHI] patient diagnosis CSV not found: {diag_path}")

    # Robust read: handle headerless patient_diagnosis.csv
    diag = pd.read_csv(diag_path)

    def _norm_col(c: object) -> str:
        return str(c).strip().lower()

    diag.columns = [_norm_col(c) for c in diag.columns]
    pid_col = next((c for c in diag.columns if ("patient" in c) or (c in ["id","patient_id","patient_number"])), None)
    dx_col  = next((c for c in diag.columns if ("diagn" in c) or (c in ["class","label"])), None)

    if pid_col is None or dx_col is None:
        diag2 = pd.read_csv(diag_path, header=None)
        if diag2.shape[1] < 2:
            for sep in [";", "\t", "|"]:
                try:
                    tmp = pd.read_csv(diag_path, header=None, sep=sep)
                    if tmp.shape[1] >= 2:
                        diag2 = tmp
                        break
                except Exception:
                    pass
        if diag2.shape[1] < 2:
            raise ValueError(f"[ICBHI] Could not parse diagnosis CSV into 2 columns: {diag_path}")
        diag = diag2.iloc[:, :2].copy()
        diag.columns = ["patient_id", "diagnosis"]
        pid_col, dx_col = "patient_id", "diagnosis"

    def _norm_pid(x: object) -> str:
        s = str(x).strip()
        if re.match(r"^\d+\.0$", s):
            s = s[:-2]
        return s

    pid_to_dx = {_norm_pid(r[pid_col]): str(r[dx_col]).strip() for _, r in diag.iterrows()}

    rows: List[Dict[str, Any]] = []
    wavs = sorted(audio_dir.glob("*.wav"))

    for wav in wavs:
        subj = _safe_subject_from_icbhi_filename(wav.name)
        dx = pid_to_dx.get(str(subj), "Unknown")
        label = 1 if str(dx).strip().lower() == "copd" else 0

        txt = wav.with_suffix(".txt")
        if not txt.exists():
            rows.append(dict(
                sample_id=f"ICBHI_{wav.stem}_full",
                dataset="ICBHI",
                subject_id=str(subj),
                filepath=str(wav),
                label=int(label),
                diagnosis=str(dx),
                start_sec=0.0,
                end_sec=-1.0,
            ))
            continue

        with open(txt, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    start = float(parts[0]); end = float(parts[1])
                except ValueError:
                    continue
                rows.append(dict(
                    sample_id=f"ICBHI_{wav.stem}_cycle{i:03d}",
                    dataset="ICBHI",
                    subject_id=str(subj),
                    filepath=str(wav),
                    label=int(label),
                    diagnosis=str(dx),
                    start_sec=start,
                    end_sec=end,
                ))

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df

def build_fraiwan_binary(
    fraiwan_root: Path,
    out_csv: Path,
    audio_subdir: str = "Audio Files",
    diag_filename: str = "Data annotation.xlsx",
    window_sec: float = 5.0,
    hop_sec: float = 2.5,
) -> pd.DataFrame:
    """
    Fraiwan annotations vary by release.
    Robust strategy:
      1) Load ALL sheets; prefer a sheet containing a filename/audio column OR cells ending with .wav.
      2) If still no explicit filename exists, fallback to row-index mapping against sorted audio files.
         (Some releases follow 'row index -> filename' convention.)
    """
    audio_dir = fraiwan_root / audio_subdir
    diag_path = fraiwan_root / diag_filename

    if not audio_dir.exists():
        raise FileNotFoundError(f"[Fraiwan] AUDIO_DIR not found: {audio_dir}")
    if not diag_path.exists():
        raise FileNotFoundError(f"[Fraiwan] Annotation file not found: {diag_path}")

    # list audio files
    audio_files = []
    for ext in ["*.wav", "*.WAV", "*.flac", "*.FLAC", "*.mp3", "*.MP3"]:
        audio_files.extend(list(audio_dir.glob(ext)))
    audio_files = sorted(audio_files, key=lambda p: p.name.lower())
    if len(audio_files) == 0:
        raise FileNotFoundError(f"[Fraiwan] No audio files found under: {audio_dir}")

    sheets = pd.read_excel(diag_path, sheet_name=None)
    if not isinstance(sheets, dict) or len(sheets) == 0:
        raise ValueError(f"[Fraiwan] Could not read any sheets from: {diag_path}")

    def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        # drop fully empty columns
        keep=[]
        for c in df.columns:
            if df[c].isna().all():
                continue
            keep.append(c)
        return df[keep]

    def find_filename_column(df: pd.DataFrame) -> str | None:
        cols = [str(c).strip() for c in df.columns]
        # header heuristic
        for c in cols:
            cl = c.lower()
            if any(k in cl for k in ["file", "filename", "audio", "recording", "path", "wav"]):
                return c
        # cell heuristic
        for c in cols:
            s = df[c].astype(str)
            if s.str.contains(r"\.wav$", case=False, na=False).any():
                return c
        return None

    def sheet_score(df: pd.DataFrame) -> int:
        c = find_filename_column(df)
        if c is not None:
            return 1000
        # count how many wav-like cells exist
        return int(df.astype(str).apply(lambda col: col.str.contains(r"\.wav", case=False, na=False)).sum().sum())

    best_name = None
    best_df = None
    best_sc = -1
    for name, df in sheets.items():
        df2 = normalize_cols(df)
        sc = sheet_score(df2)
        if sc > best_sc:
            best_sc = sc
            best_name, best_df = name, df2

    ann = best_df
    ann_sheet = best_name
    ann_cols = [str(c).strip() for c in ann.columns]

    def find_col(keys):
        for k in keys:
            for c in ann_cols:
                if k.lower() in c.lower():
                    return c
        return None

    file_col = find_filename_column(ann)
    pid_col  = find_col(["patient", "subject", "id"])
    dx_col   = find_col(["diagn", "label", "class", "disease"])
    if dx_col is None and "Diagnosis" in ann.columns:
        dx_col = "Diagnosis"

    rows: List[Dict[str, Any]] = []

    if file_col is None:
        # Fallback: row-index mapping to sorted files
        n_rows = len(ann)
        n_files = len(audio_files)
        n = min(n_rows, n_files)
        print(f"[Fraiwan] WARNING: No filename column found in sheet '{ann_sheet}'.")
        print(f"[Fraiwan] Falling back to row-index mapping: rows={n_rows}, files={n_files}, using n={n}.")
        for i in range(n):
            wav_path = audio_files[i]
            r = ann.iloc[i]
            subject = str(r[pid_col]).strip() if pid_col and pd.notna(r.get(pid_col)) else str(i)
            dx = str(r[dx_col]).strip() if dx_col and pd.notna(r.get(dx_col)) else "Unknown"
            label = 1 if str(dx).strip().lower() == "copd" else 0
            rows.append(dict(
                sample_id=f"FRAIWAN_{wav_path.stem}_full",
                dataset="FRAIWAN",
                subject_id=str(subject),
                filepath=str(wav_path),
                label=int(label),
                diagnosis=str(dx),
                start_sec=0.0,
                end_sec=-1.0,
                window_sec=float(window_sec),
                hop_sec=float(hop_sec),
                fraiwan_sheet=str(ann_sheet),
                fraiwan_row_index=int(i),
            ))
    else:
        # Normal: filename exists
        for idx, r in ann.iterrows():
            fname = str(r[file_col]).strip()
            if not fname or fname.lower() == "nan":
                continue
            wav_path = audio_dir / fname
            if not wav_path.exists():
                wav_path = Path(fname)
            if not wav_path.exists():
                stem = Path(fname).stem
                candidates = [p for p in audio_files if p.stem == stem]
                if candidates:
                    wav_path = candidates[0]
                else:
                    continue

            subject = str(r[pid_col]).strip() if pid_col and pd.notna(r.get(pid_col)) else wav_path.stem
            dx = str(r[dx_col]).strip() if dx_col and pd.notna(r.get(dx_col)) else "Unknown"
            label = 1 if str(dx).strip().lower() == "copd" else 0

            rows.append(dict(
                sample_id=f"FRAIWAN_{wav_path.stem}_full",
                dataset="FRAIWAN",
                subject_id=str(subject),
                filepath=str(wav_path),
                label=int(label),
                diagnosis=str(dx),
                start_sec=0.0,
                end_sec=-1.0,
                window_sec=float(window_sec),
                hop_sec=float(hop_sec),
                fraiwan_sheet=str(ann_sheet),
                fraiwan_row_index=int(idx),
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "[Fraiwan] No rows produced. Annotation does not map to audio files. "
            f"Audio files found={len(audio_files)}. Annotation columns={ann_cols}. Sheet='{ann_sheet}'."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df

def pooled_manifest(icbhi_csv: Path, fraiwan_csv: Path, out_csv: Path) -> pd.DataFrame:
    a = pd.read_csv(icbhi_csv)
    b = pd.read_csv(fraiwan_csv)
    df = pd.concat([a, b], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df

def diagnosis_breakdown(manifest_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(manifest_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_dx = df.groupby(["dataset", "diagnosis"]).agg(
        segments=("sample_id", "count"),
        subjects=("subject_id", "nunique"),
    ).reset_index()
    by_dx.to_csv(out_dir / "diagnosis_breakdown_by_dataset.csv", index=False)

    by_label = df.groupby(["dataset", "label"]).agg(
        segments=("sample_id", "count"),
        subjects=("subject_id", "nunique"),
    ).reset_index()
    by_label.to_csv(out_dir / "binary_breakdown_by_dataset.csv", index=False)
