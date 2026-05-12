"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier: StandardScaler → PCA → linear logistic probe.

    PCA dimension and ``C`` are chosen by **validation AUROC** when
    ``fit_hyperparameters`` runs after ``fit`` (k-fold evaluation).

    When only ``fit`` is called (final ``solution.py`` fit on train∪val),
    the same grid is scored with **stratified inner CV** (mean AUROC across
    folds), then refit on all samples.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: nn.Linear | None = None
        self._pca: PCA | None = None
        self._scaler = StandardScaler()
        self._threshold: float = 0.5

        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def _build_network(self, input_dim: int) -> None:
        self._net = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def _max_pca_components(self, n_samples: int, n_features: int) -> int:
        return int(min(max(n_samples - 1, 1), n_features, 512))

    def _pca_component_grid(self, n_samples: int, n_features: int) -> list[int]:
        cap = self._max_pca_components(n_samples, n_features)
        raw = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
        out = [k for k in raw if 8 <= k <= cap]
        if not out:
            out = [min(cap, 8)]
        return sorted(set(out))

    def _c_grid(self) -> np.ndarray:
        return np.logspace(-2.25, 0.35, 10)

    def _logistic(self, C: float) -> LogisticRegression:
        return LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=8000,
            class_weight="balanced",
            tol=1e-4,
            random_state=42,
        )

    def _copy_lr_to_linear(self, clf: LogisticRegression, input_dim: int) -> None:
        self._build_network(input_dim)
        coef = torch.from_numpy(clf.coef_.astype(np.float32))
        bias = torch.from_numpy(clf.intercept_.astype(np.float32))
        with torch.no_grad():
            self._net.weight.copy_(coef)
            self._net.bias.copy_(bias)

    def _fit_pca_lr_refit(
        self,
        X_scaled_train: np.ndarray,
        y: np.ndarray,
        n_components: int,
        C: float,
    ) -> tuple[PCA, LogisticRegression]:
        pca = PCA(n_components=n_components, random_state=42, svd_solver="full")
        Xp = pca.fit_transform(X_scaled_train)
        clf = self._logistic(C)
        clf.fit(Xp, y)
        return pca, clf

    def _select_and_fit_with_validation(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        assert self._X_train is not None and self._y_train is not None
        Xs = self._scaler.transform(self._X_train)
        y = self._y_train
        Xvs = self._scaler.transform(X_val)
        n_samples, n_features = Xs.shape

        best_auc = -1.0
        best_n: int | None = None
        best_c: float | None = None

        for n_comp in self._pca_component_grid(n_samples, n_features):
            for C in self._c_grid():
                pca = PCA(n_components=n_comp, random_state=42, svd_solver="full")
                try:
                    Xp = pca.fit_transform(Xs)
                    Xpv = pca.transform(Xvs)
                except Exception:
                    continue
                clf = self._logistic(C)
                clf.fit(Xp, y)
                probs = clf.predict_proba(Xpv)[:, 1]
                try:
                    auc = roc_auc_score(y_val, probs)
                except ValueError:
                    continue
                if auc > best_auc:
                    best_auc = auc
                    best_n = n_comp
                    best_c = float(C)

        if best_n is None or best_c is None:
            best_n = min(128, self._max_pca_components(n_samples, n_features))
            best_c = 0.1

        pca, clf = self._fit_pca_lr_refit(Xs, y, best_n, best_c)
        self._pca = pca
        self._copy_lr_to_linear(clf, best_n)
        self.eval()

    def _select_and_fit_inner_cv(self) -> None:
        assert self._X_train is not None and self._y_train is not None
        Xs = self._scaler.transform(self._X_train)
        y = self._y_train
        n_samples, n_features = Xs.shape

        n_splits = min(5, max(2, n_samples // 40))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        best_mean = -1.0
        best_n: int | None = None
        best_c: float | None = None

        for n_comp in self._pca_component_grid(n_samples, n_features):
            for C in self._c_grid():
                fold_aucs: list[float] = []
                for tr_idx, va_idx in skf.split(Xs, y):
                    X_tr, X_va = Xs[tr_idx], Xs[va_idx]
                    y_tr, y_va = y[tr_idx], y[va_idx]
                    pca = PCA(n_components=n_comp, random_state=42, svd_solver="full")
                    try:
                        Xptr = pca.fit_transform(X_tr)
                        Xpva = pca.transform(X_va)
                    except Exception:
                        fold_aucs = []
                        break
                    clf = self._logistic(C)
                    clf.fit(Xptr, y_tr)
                    probs = clf.predict_proba(Xpva)[:, 1]
                    try:
                        fold_aucs.append(roc_auc_score(y_va, probs))
                    except ValueError:
                        pass
                if len(fold_aucs) < n_splits:
                    continue
                m = float(np.mean(fold_aucs))
                if m > best_mean:
                    best_mean = m
                    best_n = n_comp
                    best_c = float(C)

        if best_n is None or best_c is None:
            best_n = min(128, self._max_pca_components(n_samples, n_features))
            best_c = 0.1

        pca, clf = self._fit_pca_lr_refit(Xs, y, best_n, best_c)
        self._pca = pca
        self._copy_lr_to_linear(clf, best_n)
        self.eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit scaler and cache training data; PCA/LR are fit later."""
        self._X_train = np.asarray(X, dtype=np.float64)
        self._y_train = np.asarray(y, dtype=np.int64)
        self._scaler.fit(self._X_train)
        self._pca = None
        self._net = None
        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Pick PCA dim + C by validation AUROC; tune threshold on validation accuracy."""
        self._select_and_fit_with_validation(
            np.asarray(X_val, dtype=np.float64),
            np.asarray(y_val, dtype=np.int64),
        )

        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 501)])
        )

        best_threshold = 0.5
        best_acc = -1.0
        best_bal = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            acc = accuracy_score(y_val, y_pred_t)
            bal = balanced_accuracy_score(y_val, y_pred_t)
            if acc > best_acc or (
                np.isclose(acc, best_acc) and bal > best_bal
            ):
                best_acc = acc
                best_bal = bal
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None or self._pca is None:
            if self._X_train is None:
                raise RuntimeError("Call fit() before predict_proba().")
            self._select_and_fit_inner_cv()

        assert self._pca is not None and self._net is not None
        X_scaled = self._scaler.transform(np.asarray(X, dtype=np.float64))
        X_pca = self._pca.transform(X_scaled).astype(np.float32)
        X_t = torch.from_numpy(X_pca)
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
