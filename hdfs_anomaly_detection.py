#!/usr/bin/env python3
"""HDFS Log Anomaly Detection a complete, standalone experiment.

Install: python -m pip install numpy pandas scikit-learn matplotlib torch
Demo:    python hdfs_anomaly_detection.py --synthetic
HDFS:    python hdfs_anomaly_detection.py --data-dir data

Put HDFS.log and anomaly_label.csv in data/ for the real-data run.
Outputs: results.csv, confusion matrices, plots, and experiment.json.

Methodology
-----------
Sessions are split 60/20/20 before fitting the event vocabulary. A stateless
regex parser removes metadata and variable identifiers; it is deliberately
used instead of an online template miner to avoid fitting on held-out logs.
Both detectors train on normal training sessions. Validation labels select
F1-optimal thresholds; test labels are used only for final evaluation.
Isolation Forest discards order. The LSTM reconstructs ordered token sequences
through a latent bottleneck. Padding is excluded from encoding and loss.

Limitations: regex normalization is simpler than Drain3; random block splits
are not temporal deployment tests; truncation loses long-session information;
validation F1 tuning assumes labeled anomalies; synthetic scores do not measure
performance on HDFS. An autoencoder can also reconstruct some anomalies well.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

BLOCK_RE = re.compile(r"blk_-?\d+")
HEADER_RE = re.compile(r"^\d{6}\s+\d{6}\s+\d+\s+\w+\s+\S+:?\s+")


def normalize_line(line):
    """Normalize real HDFS metadata, IDs, addresses, paths and numbers."""
    content = HEADER_RE.sub("", line.strip())
    content = BLOCK_RE.sub("<BLOCK>", content)
    content = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?", "<IP>", content)
    content = re.sub(r"/(?:[\w.\-]+/)*[\w.\-]+", "<PATH>", content)
    content = re.sub(r"\b\d+\b", "<NUM>", content)
    return content


def synthetic_sessions(seed, n_normal=8000, n_anomaly=240):
    rng = random.Random(seed)
    rows = []
    for i in range(n_normal + n_anomaly):
        seq = ["allocate"] + ["receive", "write", "ack", "close"] * rng.choice([2, 3, 3]) + ["store", "verify"]
        if rng.random() < .3:
            seq.append("verify")
        anomalous = i >= n_normal
        if anomalous:
            kind = rng.choice(["shuffle", "missing", "error", "extra"])
            if kind == "shuffle":
                rng.shuffle(seq)
            elif kind == "missing":
                seq = [e for e in seq if rng.random() > .4] or ["receive"]
            elif kind == "error":
                seq = seq[:3] + ["io_exception", "replication_timeout"]
            else:
                seq += ["io_exception"] * rng.randint(2, 5)
        rows.append({"BlockId": f"blk_{i}", "Sequence": seq, "Label": int(anomalous)})
    return pd.DataFrame(rows)


def load_sessions(data_dir):
    log_path, label_path = data_dir / "HDFS.log", data_dir / "anomaly_label.csv"
    for path in (log_path, label_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Supply HDFS files or use --synthetic explicitly.")
    labels = pd.read_csv(label_path, dtype={"BlockId": str})
    if not {"BlockId", "Label"}.issubset(labels.columns):
        raise ValueError("Labels must contain BlockId and Label columns.")
    if labels.BlockId.duplicated().any():
        raise ValueError("Duplicate block IDs in labels.")
    mapping = labels.Label.astype(str).str.strip().str.lower().map({"normal": 0, "anomaly": 1})
    if mapping.isna().any():
        raise ValueError("Labels must be Normal or Anomaly.")
    label_map = dict(zip(labels.BlockId, mapping.astype(int)))
    sequences = defaultdict(list)
    lines = 0
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines += 1
            # A repeated mention of a block within one log line is one event.
            block_ids = set(BLOCK_RE.findall(line))
            if block_ids:
                event = normalize_line(line)
                for block in block_ids:
                    if block in label_map:
                        sequences[block].append(event)
    print(f"Read {lines:,} lines; {len(sequences):,} labeled sessions; "
          f"{len(labels) - len(sequences):,} labels without logs.")
    return pd.DataFrame([{"BlockId": b, "Sequence": s, "Label": label_map[b]}
                         for b, s in sequences.items()])


def split_sessions(df, seed):
    if df.empty or df.Label.nunique() != 2 or df.Label.value_counts().min() < 10:
        raise ValueError("Need both classes and at least 10 sessions per class for stratified splits.")
    train, held = train_test_split(df, test_size=.4, stratify=df.Label, random_state=seed)
    val, test = train_test_split(held, test_size=.5, stratify=held.Label, random_state=seed)
    return [x.reset_index(drop=True) for x in (train, val, test)]


def count_features(sequences, vocabulary):
    """Last column counts events unseen in normal training sessions."""
    result = np.zeros((len(sequences), len(vocabulary) + 1), dtype=np.float32)
    for row, seq in enumerate(sequences):
        for event in seq:
            result[row, vocabulary.get(event, len(vocabulary))] += 1
    return result


def best_threshold(scores, labels):
    """O(n log n) F1 selection; ties favor the higher threshold."""
    scores = np.asarray(scores)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be nonempty and finite.")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[index]), float(f1[index])


def metrics(name, labels, predictions, scores=None):
    p, r, f, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {"Model": name, "Precision": p, "Recall": r, "F1": f,
            "Accuracy": accuracy_score(labels, predictions),
            "AveragePrecision": average_precision_score(labels, scores) if scores is not None else np.nan}


def encode_sequences(sequences, vocabulary, max_len):
    x = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        encoded = [vocabulary.get(e, 1) for e in seq[:max_len]]
        x[i, :len(encoded)] = encoded
    return x


def fit_lstm(normal, val, test, vocabulary, max_len, args):
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.nn.utils.rnn import pack_padded_sequence
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Install PyTorch: python -m pip install torch (or use --skip-lstm)") from exc
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = len(vocabulary) + 2

    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(size, 16, padding_idx=0)
            self.encoder = nn.LSTM(16, 32, batch_first=True)
            self.decoder = nn.LSTM(32, 32, batch_first=True)
            self.output = nn.Linear(32, size)

        def forward(self, x):
            lengths = (x != 0).sum(1).cpu()
            packed = pack_padded_sequence(self.embedding(x), lengths, batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.encoder(packed)
            latent = hidden[-1].unsqueeze(1).expand(-1, x.size(1), -1)
            decoded, _ = self.decoder(latent)
            return self.output(decoded)

    model = LSTMAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    normal_x = torch.from_numpy(encode_sequences(normal.Sequence, vocabulary, max_len))
    loader = DataLoader(TensorDataset(normal_x), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    losses = []
    for epoch in range(args.epochs):
        model.train()
        total, tokens = 0., 0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss_sum = F.cross_entropy(logits.reshape(-1, size), x.reshape(-1), ignore_index=0, reduction="sum")
            count = (x != 0).sum()
            (loss_sum / count).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss_sum.item()
            tokens += count.item()
        losses.append(total / tokens)
        print(f"LSTM epoch {epoch + 1}/{args.epochs}: token loss {losses[-1]:.4f}", flush=True)

    @torch.no_grad()
    def score(frame):
        model.eval()
        x_all = torch.from_numpy(encode_sequences(frame.Sequence, vocabulary, max_len))
        scores = []
        for (x,) in DataLoader(TensorDataset(x_all), batch_size=args.batch_size):
            x = x.to(device)
            token_loss = F.cross_entropy(model(x).reshape(-1, size), x.reshape(-1), ignore_index=0, reduction="none").reshape_as(x)
            scores.append((token_loss.sum(1) / (x != 0).sum(1)).cpu().numpy())
        return np.concatenate(scores)
    return score(val), score(test), losses, str(device)


def plot_scores(name, scores, labels, threshold, output):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.histogram_bin_edges(scores, bins=40)
    for label, text in [(0, "Normal"), (1, "Anomaly")]:
        ax.hist(scores[labels == label], bins=bins, density=True, alpha=.5, label=text)
    ax.axvline(threshold, color="black", linestyle="--", label="Validation threshold")
    ax.set(title=f"{name}: validation scores", xlabel="Anomaly score", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--synthetic", action="store_true", help="Explicit demo mode; not a real HDFS benchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-lstm", action="store_true", help="Run only the classical baseline")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.max_len, args.threads) < 1:
        parser.error("Epochs, batch size, maximum length and threads must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mode = "SYNTHETIC DEMO" if args.synthetic else "HDFS"
    print(f"Data mode: {mode}")
    df = synthetic_sessions(args.seed) if args.synthetic else load_sessions(args.data_dir)
    train, val, test = split_sessions(df, args.seed)
    normal = train.loc[train.Label == 0].reset_index(drop=True)
    for name, frame in [("Train", train), ("Validation", val), ("Test", test)]:
        print(f"{name}: {len(frame):,} sessions; anomaly rate {frame.Label.mean():.2%}")
    print(f"Normal training sessions: {len(normal):,}")
    vocabulary = {event: i for i, event in enumerate(sorted({e for s in normal.Sequence for e in s}))}
    y_val, y_test = val.Label.to_numpy(), test.Label.to_numpy()
    iso = IsolationForest(n_estimators=200, random_state=args.seed, contamination="auto", n_jobs=-1)
    iso.fit(count_features(normal.Sequence, vocabulary))
    scores = {"Isolation Forest": (-iso.score_samples(count_features(val.Sequence, vocabulary)),
                                   -iso.score_samples(count_features(test.Sequence, vocabulary)))}
    metadata = {"data_mode": mode, "seed": args.seed, "normal_training_sessions": len(normal),
                "split_sizes": dict(train=len(train), validation=len(val), test=len(test)),
                "normal_training_event_types": len(vocabulary), "thresholds": {}}
    lengths = df.Sequence.map(len)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df.Label.value_counts().reindex([0, 1], fill_value=0).rename(index={0: "Normal", 1: "Anomaly"}).plot.bar(ax=axes[0])
    axes[0].set(title=f"Class distribution ({mode})", ylabel="Sessions")
    axes[1].hist(lengths, bins=40)
    axes[1].set(title="Session lengths", xlabel="Events", ylabel="Sessions")
    fig.tight_layout(); fig.savefig(output / "eda.png", dpi=150); plt.close(fig)
    if not args.skip_lstm:
        seq_vocab = {e: i + 2 for e, i in vocabulary.items()}
        max_len = min(args.max_len, int(normal.Sequence.map(len).max()))
        metadata["max_sequence_length"] = max_len
        metadata["truncated_sessions"] = {name: int(frame.Sequence.map(len).gt(max_len).sum())
                                           for name, frame in [("train", train), ("validation", val), ("test", test)]}
        v, t, losses, device = fit_lstm(normal, val, test, seq_vocab, max_len, args)
        scores["LSTM Autoencoder"] = (v, t)
        metadata["device"] = device
        metadata["epochs"] = args.epochs
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(losses) + 1), losses)
        ax.set(xlabel="Epoch", ylabel="Mean token cross entropy", title="Normal-session training loss")
        fig.tight_layout(); fig.savefig(output / "training_loss.png", dpi=150); plt.close(fig)
    rows = [metrics("Always Normal", y_test, np.zeros_like(y_test))]
    for name, (val_scores, test_scores) in scores.items():
        threshold, val_f1 = best_threshold(val_scores, y_val)
        pred = (test_scores >= threshold).astype(int)
        rows.append(metrics(name, y_test, pred, test_scores))
        slug = name.lower().replace(" ", "_")
        metadata["thresholds"][name] = {"threshold": threshold, "validation_f1": val_f1}
        pd.DataFrame(confusion_matrix(y_test, pred, labels=[0, 1]), index=["True Normal", "True Anomaly"],
                     columns=["Pred Normal", "Pred Anomaly"]).to_csv(output / f"{slug}_confusion.csv")
        pd.DataFrame({"BlockId": test.BlockId, "Label": y_test, "Score": test_scores,
                      "Prediction": pred}).to_csv(output / f"{slug}_test_predictions.csv", index=False)
        plot_scores(name, val_scores, y_val, threshold, output / f"{slug}_scores.png")
    results = pd.DataFrame(rows).set_index("Model")
    results.to_csv(output / "results.csv")
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nTest results — {mode}\n{results.round(4).to_string()}")
    print(f"\nOutputs saved to {output.resolve()}")


if __name__ == "__main__":
    main()
