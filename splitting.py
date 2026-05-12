"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    Uses stratified K-fold: each fold holds out a disjoint test chunk; the
    remainder is split into train / validation (stratified) for threshold tuning.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Unused (kept for API compatibility).
        val_size:     Fraction of the *non-test* pool used for validation.
        random_state: Random seed for reproducible splits.
        n_splits:     Number of stratified folds.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    # Hold-out test per fold is ~1/K of data; scale val fraction so ~``val_size``
    # of the full dataset sits in validation (same intent as the old single split).
    val_frac_of_tv = val_size * n_splits / max(n_splits - 1, 1)
    val_frac_of_tv = float(min(0.45, max(0.08, val_frac_of_tv)))

    for train_val_idx, idx_test in skf.split(idx, y):
        y_tv = y[train_val_idx]
        idx_train, idx_val = train_test_split(
            train_val_idx,
            test_size=val_frac_of_tv,
            random_state=random_state,
            stratify=y_tv,
        )
        splits.append((idx_train, idx_val, idx_test))

    return splits

