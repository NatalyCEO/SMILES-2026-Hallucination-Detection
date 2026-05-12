"""
Hallucination Detection in Small Language Models

# Files you can edit:
    - aggregation.py — layer selection and token pooling 
    - aggregation.py | extract_geometric_features — optional hand-crafted features 
    - probe.py | HallucinationProbe — probe classifier (nn.Module subclass) 
    - splitting.py | split_data — train / validation / test split strategy 

# Fixed infrastructure (do not edit)
    - model.py | LLM loader (get_model_and_tokenizer) 
    - evaluate.py | Evaluation loop, summary table, JSON output 

# Data Format — ChatML and Special Tokens
    The `prompt` column uses ChatML (Chat Markup Language), the conversation
    template built into Qwen models.  Each message is wrapped in role markers:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    ... question and context ... <|im_end|>
    <|im_start|>assistant

    Special tokens and their roles:

    - `<|im_start|>` — opens a chat turn; the role (`system`, `user`, or `assistant`) immediately follows
    - `<|im_end|>` — closes the current chat turn
    - `<|endoftext|>` — end-of-sequence (EOS) token appended by the model at the end of its response

    The `prompt` ends right after `<|im_start|>assistant\n` — it provides the
    full context up to (but not including) the model's reply.  The `response`
    column holds the actual generated text, ending with `<|endoftext|>`.

    We feed the concatenation of `prompt + response` to the feature extractor
    so the hidden states capture both the question context and the model's
    specific answer — the hallucination signal lives in that joint representation.


"""

import gc
import os
import time

# Stabilise long CPU runs (OpenMP/MKL + allocator fragmentation).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aggregation import aggregation_and_feature_extraction
from evaluate import print_summary, run_evaluation, save_predictions, save_results
from model import MAX_LENGTH, _DEFAULT_MODEL
from probe import HallucinationProbe
from splitting import split_data

# ---------------------------------------------------------------------

DATA_FILE     = "./data/dataset.csv"   # path to the dataset CSV
OUTPUT_FILE   = "results.json"         # where to write the results summary
BATCH_SIZE    = 1  # smaller batches reduce peak RAM (lm_head logits scale with batch size)
USE_GEOMETRIC = False                  # set True to enable geometric feature extraction
TEST_FILE        = "./data/test.csv"   # competition test set (labels are null)
PREDICTIONS_FILE = "predictions.csv"   # output file with predicted labels

assert OUTPUT_FILE == "results.json"
assert PREDICTIONS_FILE == "predictions.csv"

# ---------------------------------------------------------------------
if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.set_num_threads(1)

    def load_llm():
        """Load like model.get_model_and_tokenizer; CPU uses float32 + stable attention (Win)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[Model] Loading '{_DEFAULT_MODEL}' ...")
        tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_MODEL)
        dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        extra = {}
        if device.type == "cpu":
            extra["low_cpu_mem_usage"] = True
            extra["attn_implementation"] = "eager"
        m = AutoModelForCausalLM.from_pretrained(
            _DEFAULT_MODEL,
            output_hidden_states=True,
            dtype=dtype,
            **extra,
        )
        m.eval()
        return m, tokenizer

    print(f"Device       : {device}")
    print(f"Data         : {DATA_FILE}")
    print(f"Max length   : {MAX_LENGTH} tokens")
    print(f"Geometric feats: {USE_GEOMETRIC}")


    df = pd.read_csv(DATA_FILE)

    # Build the text fed to the LLM: concatenation of prompt and response.
    all_texts  = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    all_labels = np.array([int(float(h)) for h in df["label"]])

    n_total = len(all_labels)
    print(f"Loaded {n_total} samples  "
        f"({all_labels.sum()} hallucinated / {(all_labels == 0).sum()} truthful)")
    
    # Preview the raw data
    print(f"Columns : {df.columns.tolist()}")
    print(f"Rows    : {len(df)}")
    print(f"Labels  : {dict(df['label'].value_counts().sort_index())}")
    print()

    # Show the first sample (truncated for readability)
    row0 = df.iloc[0]
    print("── prompt (first 500 chars) " + "─" * 34)
    print(row0["prompt"][:500])
    print()
    print("── response (first 300 chars) " + "─" * 31)
    print(row0["response"][:300])
    print()
    label_str = "hallucinated" if int(row0["label"]) else "truthful"
    print(f"── label : {int(row0['label'])}  ({label_str})")


    # ── Train features (in memory only; no disk checkpoints) ────────────────
    model, tokenizer = load_llm()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    t0_extract = time.time()
    pending_blocks: list[np.ndarray] = []

    for start in tqdm(
        range(0, len(all_texts), BATCH_SIZE),
        desc="Extracting & aggregating",
        unit="batch",
    ):
        batch_texts = all_texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()

        batch_mat = np.vstack(
            [
                aggregation_and_feature_extraction(
                    hidden[i],
                    mask[i],
                    use_geometric=USE_GEOMETRIC,
                )
                .cpu()
                .numpy()
                for i in range(hidden.size(0))
            ]
        )
        pending_blocks.append(batch_mat)

        del outputs, hidden, input_ids, attention_mask, encoding
        gc.collect()

    extract_time = time.time() - t0_extract
    X = np.vstack(pending_blocks)
    print(
        f"Train extraction done in {extract_time:.1f} s — "
        f"{X.shape[0]} × {X.shape[1]}"
    )

    del model, tokenizer
    model, tokenizer = None, None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y = all_labels

    print(f"Feature matrix : {X.shape}  (feature_dim = {X.shape[1]})")
    print(f"Geometric feats: {USE_GEOMETRIC}")

    splits = split_data(y, df)

    print(f"Splits : {len(splits)} fold(s)")
    for i, (tr, va, te) in enumerate(splits):
        print(f"  Fold {i + 1}: train={len(tr)}  "
            f"val={len(va) if va is not None else 'N/A'}  test={len(te)}")

    fold_results = run_evaluation(splits, X, y, HallucinationProbe)
    
    print_summary(fold_results, X.shape[1], len(X), extract_time)
    save_results(fold_results, X.shape[1], len(X), extract_time, OUTPUT_FILE)

    

    # ── Load test data ────────────────────────────────────────────────────────
    df_test    = pd.read_csv(TEST_FILE)
    test_texts = [f"{row['prompt']}{row['response']}" for _, row in df_test.iterrows()]
    test_ids   = df_test.index
    n_test = len(test_texts)
    print(f"Test set loaded: {n_test} samples")

    # ── Test features (in memory only) ──────────────────────────────────────
    model, tokenizer = load_llm()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    test_blocks: list[np.ndarray] = []
    for start in tqdm(
        range(0, n_test, BATCH_SIZE),
        desc="Test extraction & aggregation",
        unit="batch",
    ):
        batch_texts = test_texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()

        batch_mat = np.vstack(
            [
                aggregation_and_feature_extraction(
                    hidden[i], mask[i], use_geometric=USE_GEOMETRIC
                )
                .cpu()
                .numpy()
                for i in range(hidden.size(0))
            ]
        )
        test_blocks.append(batch_mat)

        del outputs, hidden, input_ids, attention_mask, encoding
        gc.collect()

    X_test = np.vstack(test_blocks)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Fit final probe on training + validation data only ──────────────────
    # Collect the union of all train and validation indices across every split.
    # For a single split this excludes idx_test; for k-fold every sample appears
    # in a training fold, so all samples are used (same as fitting on X, y).
    idx_non_test = np.unique(np.concatenate([
        np.concatenate([idx_tr, idx_va]) if idx_va is not None else idx_tr
        for idx_tr, idx_va, _ in splits
    ]))
    final_probe = HallucinationProbe()
    final_probe.fit(X[idx_non_test], y[idx_non_test])

    # ── Predict and save ────────────────────────────────────────────────────
    save_predictions(final_probe, X_test, test_ids, PREDICTIONS_FILE)

