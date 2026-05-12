"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


def _l2_normalize(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vec / vec.norm(p=2).clamp(min=eps)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # Layer 0 = embeddings; transformer layers are indices 1 .. n_layers-1.
    dev = hidden_states.device
    attention_mask = attention_mask.to(dev)
    n_layers = hidden_states.size(0)
    seq_len = int(attention_mask.sum().item())
    nz = attention_mask.nonzero(as_tuple=False)
    real_positions = nz.squeeze(-1) if nz.dim() > 1 else nz
    last_pos = int(real_positions[-1].item())

    tr_layers = list(range(1, n_layers))
    idx_pick = [
        tr_layers[len(tr_layers) // 6],
        tr_layers[len(tr_layers) // 3],
        tr_layers[len(tr_layers) // 2],
        tr_layers[2 * len(tr_layers) // 3],
        tr_layers[-1],
    ]

    # L2-normalise multi-layer last-token vectors so shallow/deep scales match.
    last_token_feats = [
        _l2_normalize(hidden_states[i, last_pos].float()) for i in idx_pick
    ]

    final_layer = hidden_states[-1].float()
    mask_f = attention_mask.float().unsqueeze(-1)
    denom = mask_f.sum().clamp(min=1.0)
    mean_seq = (final_layer * mask_f).sum(dim=0) / denom

    # Exponential position weights emphasise the tail (assistant reply region).
    full_len = hidden_states.size(1)
    pos_idx = torch.arange(full_len, device=dev, dtype=torch.float32)
    tail_w = torch.exp(3.5 * pos_idx / max(full_len - 1, 1)) * attention_mask.float()
    tail_w = tail_w / tail_w.sum().clamp(min=1e-12)
    weighted_mean = (final_layer * tail_w.unsqueeze(-1)).sum(dim=0)

    # Mean over late sequence only — ответ ассистента обычно в последних токенах.
    n_real = len(real_positions)
    cut = max(0, int(0.70 * n_real))
    suffix_idx = real_positions[cut:]
    if suffix_idx.numel() > 0:
        suffix_mean = final_layer[suffix_idx].mean(dim=0)
    else:
        suffix_mean = final_layer[last_pos].clone()

    centered = final_layer - mean_seq.unsqueeze(0)
    std_seq = torch.sqrt((centered.pow(2) * mask_f).sum(dim=0) / denom)

    # Window over last few tokens (short answers stay informative).
    win = min(48, len(real_positions))
    win_idx = real_positions[-win:]
    tail_window_mean = final_layer[win_idx].mean(dim=0)

    lt = final_layer[last_pos]
    contrast = lt - mean_seq
    contrast_w = lt - weighted_mean
    cos_lm = (lt * mean_seq).sum() / (lt.norm(p=2) * mean_seq.norm(p=2)).clamp(min=1e-12)
    scalars = torch.tensor(
        [cos_lm.item(), float(seq_len) / 512.0],
        device=dev,
        dtype=torch.float32,
    )

    parts: list[torch.Tensor] = [
        *last_token_feats,
        mean_seq,
        weighted_mean,
        suffix_mean,
        std_seq,
        tail_window_mean,
        contrast,
        contrast_w,
        scalars,
    ]
    return torch.cat(parts, dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    attention_mask = attention_mask.to(hidden_states.device)

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0, device=hidden_states.device, dtype=torch.float32)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
