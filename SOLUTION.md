# Solution Report

## 1. Reproducibility

### Environment

- Python 3.x with packages from `requirements.txt` (`torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`).
- GPU (CUDA / Colab T4) is recommended; the pipeline also runs on CPU but is much slower.
- The checkpoint loaded from Hugging Face is `Qwen/Qwen2.5-0.5B` (see `model.py`).

### Exact commands

From the repository root:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

On **Windows**, if the console encoding breaks on Unicode banner lines in `solution.py`, set UTF-8 for the session before running:

```powershell
$env:PYTHONIOENCODING='utf-8'
python solution.py
```

Linux / macOS:

```bash
PYTHONIOENCODING=utf-8 python solution.py
```

### Outputs

Running `solution.py` must produce:

- `results.json` — evaluation summary from `evaluate.save_results`.
- `predictions.csv` — columns `id`, `label` for `data/test.csv` (from `evaluate.save_predictions`).


### Implementation details that affect reproducibility

1. **`solution.py`**  
   - `BATCH_SIZE = 1` — reduces peak RAM from the LM head logits on long sequences (needed on CPU / low-memory machines). On Colab GPU you can increase this (e.g. `4`) for speed.  
   - After each forward batch, tensors are deleted and `gc.collect()` runs to limit allocator fragmentation on long CPU jobs.  
   - `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `torch.set_num_threads(1)` — reduce flaky CPU backend errors on long runs.

2. **`agregation.py`,`probe.py` and `splitting.py`**  
   - `aggregation.py` — multi-layer + pooled features.  
   - `probe.py` — `StandardScaler` → **PCA** → linear logistic probe; **PCA dimension and C** chosen by **validation AUROC** on each outer fold (grid search); **threshold** tuned on **validation accuracy** (tie-break balanced accuracy). Final fit on all train∪val indices uses **inner CV mean AUROC** to pick PCA/C before refitting on full data.  
   - `splitting.py` — stratified **5-fold** CV for evaluation; final probe is fit on all non-test indices united across folds (as in `solution.py`).

Random seeds: `splitting.split_data(..., random_state=42)`, `sklearn` components in `probe.py` use `random_state=42` where applicable.

---

## 2. Final solution description

### Files modified

| File | Role |
|------|------|
| `aggregation.py` | Hidden-state aggregation and feature vector |
| `probe.py` | `HallucinationProbe` classifier |
| `splitting.py` | Train / validation / test splits |


### Approach

1. **Features (`aggregate`)**  
   - **Five transformer layers** — last-token vectors (L2-normalised per layer), spaced across depth.  
   - **Final-layer pooling** — masked mean; **exponential tail weights** (emphasis on the end of the sequence, assistant reply); **mean over the last ~30% of real tokens** (suffix of the sequence, where the answer usually lives) instead of max-pool; masked standard deviation over the sequence.  
   - **Tail window** — mean over the last up-to-48 real tokens.  
   - **Contrasts** — last token minus global mean; last token minus tail-weighted mean.  
   - **Scalars** — cosine similarity between last token and sequence mean; normalised sequence length.  

   Resulting dimensionality: **10754** (for hidden size 896).

2. **Probe (`HallucinationProbe`)**  
   - `StandardScaler` on inputs.  
   - **PCA** on scaled features (dimension from a fixed candidate grid, capped by sample count).  
   - **Logistic regression** on PCA coordinates: `sklearn.linear_model.LogisticRegression` (`class_weight="balanced"`, L-BFGS); **C** comes from a **log-spaced grid**, not a single constant. Weights are copied into one `nn.Linear` for `forward` / `predict_proba`.  
   - **Hyperparameters (PCA dim + `C`)**: maximised **validation AUROC** within each fold; for the final full-data fit (no held-out val in `solution.py`), **mean AUROC across inner stratified CV folds** on the training matrix.  
   - **Threshold**: `fit_hyperparameters` maximises **validation accuracy**; ties broken by **balanced accuracy**.

3. **Splits (`split_data`)**  
   - **StratifiedKFold (n_splits=5)**.  
   - Validation fraction within the non-test pool scaled so validation size is ~15% of the full dataset in intent.

### What helped the metric most

- **Tail-focused pooling** (weighted mean, suffix mean over late tokens, last-token contrasts) targets the assistant’s answer span rather than the whole prompt.  
- **PCA + linear logistic probe** (with **C grid**) lowers effective dimensionality vs raw 10754-d features and tends to generalise better than a high-dimensional linear model on small `N`.  
- **Validation AUROC** for picking PCA/C stabilises ranking quality; **threshold** tuning on validation **accuracy** aligns decisions with the leaderboard metric (accuracy on `test.csv`).  

### Local evaluation snapshot (`results.json`)


| Metric (avg over folds, probe on held-out **internal** test split) | Value |
|-------------------------------------------------------------------|-------|
| Test **accuracy** | **~0.718** |
| Test **AUROC** | **~0.711** |
| Train accuracy (avg) | **~0.823** |
| Val accuracy (avg) | **~0.740** |
| `feature_dim` | **10754** |
| `n_folds` | **5** |

The table is **cross-validation on `dataset.csv`**.

---

## 3. Experiments and failed / discarded ideas

| Idea | Outcome |
|------|---------|
| **Last layer + last token only** | Simple baseline; weaker than multi-layer + tail pooling. |
| **Tune threshold with F1** | Can improve F1 while hurting accuracy; validation **accuracy** used for threshold selection. |
| **`USE_GEOMETRIC = True`** | Optional path in `aggregation.py` not enabled in `solution.py`; main signal kept in `aggregate()`. |
| **Very deep MLP / very long training** | Risk of overfitting; superseded by **PCA + logistic** probe. |
| **Deep MLP + Adam** | Replaced by **PCA + sklearn `LogisticRegression`** mapped to `nn.Linear`. |
| **max pool over sequence** | Replaced by **mean over last ~30% of real tokens** — aligns with answer-heavy tail. |
| **Selecting PCA/C by validation accuracy** (instead of AUROC) | Tried to match the accuracy leaderboard directly; on our CV runs **mean test accuracy / AUROC did not improve**, so the repo **reverted** to **AUROC-based** selection for PCA/C. |

