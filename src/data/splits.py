from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    fold: int

def make_subjectwise_folds(
    df: pd.DataFrame,
    subject_col: str,
    label_col: str,
    n_splits: int,
    val_frac: float,
    shuffle: bool,
    seed: int,
    stratify_by_label: bool = True,
) -> List[Split]:
    # Subject-level frame: one row per subject with subject label = majority label
    subj = df[[subject_col, label_col]].copy()
    subj = subj.groupby(subject_col)[label_col].agg(lambda x: int(round(x.mean()))).reset_index()
    subjects = subj[subject_col].astype(str).values
    labels = subj[label_col].values

    if stratify_by_label:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        split_iter = kf.split(subjects, labels)
    else:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
        split_iter = kf.split(subjects)

    splits: List[Split] = []
    for fold, (trainval_subj_idx, test_subj_idx) in enumerate(split_iter):
        trainval_subjects = subjects[trainval_subj_idx]
        test_subjects = subjects[test_subj_idx]

        # split trainval subjects into train/val subjects
        if stratify_by_label:
            train_subj, val_subj = train_test_split(
                trainval_subjects,
                test_size=val_frac,
                random_state=seed + fold,
                stratify=labels[trainval_subj_idx],
            )
        else:
            train_subj, val_subj = train_test_split(
                trainval_subjects,
                test_size=val_frac,
                random_state=seed + fold,
            )

        # Map to sample indices
        train_idx = df.index[df[subject_col].astype(str).isin(set(train_subj))].to_numpy()
        val_idx   = df.index[df[subject_col].astype(str).isin(set(val_subj))].to_numpy()
        test_idx  = df.index[df[subject_col].astype(str).isin(set(test_subjects))].to_numpy()

        # sanity: disjoint subjects
        s_train = set(df.loc[train_idx, subject_col].astype(str))
        s_val   = set(df.loc[val_idx, subject_col].astype(str))
        s_test  = set(df.loc[test_idx, subject_col].astype(str))
        assert s_train.isdisjoint(s_val)
        assert s_train.isdisjoint(s_test)
        assert s_val.isdisjoint(s_test)

        splits.append(Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, fold=fold))
    return splits
