"""
python graph/build_knn_graph.py \
  --embeddings "/home/azureuser/PITL/outputs/patient_embeddings.pt" \
  --out-json "/home/azureuser/PITL/outputs/neighbors/knn_train.json" \
  --k 10

"""

import argparse
import json
import os
from pathlib import Path

import torch
import numpy as np


def load_embeddings(path: str):
    """
    Expect a .pt saved by torch.save.
    It can be:
      - a tensor [N, D], or
      - a dict with keys like 'emb', 'e', or 'embeddings' and optional 'case_ids'.
    """
    obj = torch.load(path, map_location="cpu")

    case_ids = None
    if isinstance(obj, torch.Tensor):
        emb = obj
    elif isinstance(obj, dict):
        # try a few common keys
        for key in ["emb", "embeddings", "e", "E"]:
            if key in obj:
                emb = obj[key]
                break
        else:
            raise ValueError(
                f"Could not find embeddings in {path}. "
                f"Expected one of keys ['emb', 'embeddings', 'e', 'E']."
            )

        if "case_ids" in obj:
            case_ids = [str(x) for x in obj["case_ids"]]
    else:
        raise ValueError(f"Unsupported embeddings format in {path} (type {type(obj)})")

    if emb.dim() != 2:
        raise ValueError(f"Embeddings must be [N, D], got shape {tuple(emb.shape)}")

    return emb.float(), case_ids


def compute_knn_cosine(emb: torch.Tensor, k: int):
    """
    emb: [N, D] (not necessarily normalized).
    Returns:
      indices: [N, k] neighbor indices per row
      sims:    [N, k] cosine similarity values
    """
    N, D = emb.shape
    # normalize
    eps = 1e-8
    x = emb / (emb.norm(dim=1, keepdim=True) + eps)  # [N, D]

    # cosine similarity matrix
    sims = x @ x.T  # [N, N]

    # exclude self
    diag_idx = torch.arange(N)
    sims[diag_idx, diag_idx] = -1e9

    # k nearest neighbors by similarity
    k = min(k, N - 1)
    vals, idxs = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)

    return idxs.cpu().numpy(), vals.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Build kNN graph on patient embeddings.")
    ap.add_argument("--embeddings", required=True, help="Path to .pt embeddings file")
    ap.add_argument("--out-json", required=True, help="Output JSON path")
    ap.add_argument(
        "--k", type=int, default=10, help="Number of neighbors (8-12 suggested)"
    )
    args = ap.parse_args()

    emb, case_ids = load_embeddings(args.embeddings)
    N, D = emb.shape
    print(f"[info] Loaded embeddings: N={N}, D={D}")

    idxs, sims = compute_knn_cosine(emb, k=args.k)
    print(f"[info] Computed kNN with k={args.k}")

    out = {
        "num_nodes": int(N),
        "dim": int(D),
        "k": int(args.k),
        "metric": "cosine",
        "indices": idxs.tolist(),  # shape [N, k]
        "similarities": sims.tolist(),  # shape [N, k]
    }
    if case_ids is not None:
        out["case_ids"] = list(case_ids)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"[info] Wrote kNN graph to {args.out_json}")


if __name__ == "__main__":
    main()
