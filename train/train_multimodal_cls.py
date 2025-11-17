"""
PYTHONPATH=. python train/train_multimodal_cls.py \
  --manifest-csv "/home/azureuser/PITL/pre_processing/nsclc_manifest.csv" \
  --ehr-csv "/home/azureuser/PITL/pre_processing/NSCLC_EHR_features.csv" \
  --cutoff-days 1096 \
  --batch-size 2 \
  --epochs 100 \
  --val-frac 0.2 \
  --device "cuda:0" \
  --num-workers 4 \
  --amp \
  --out-dir "/home/azureuser/PITL/outputs/multimodal_cls"
"""

import argparse
import os
import random
from pathlib import Path
from monai.data.utils import pad_list_data_collate

from torch.cuda.amp import GradScaler, autocast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.nsclc_dataset import NsclcMultimodalDataset, get_nsclc_transforms
from models.nsclc_multimodal_cls_model import NsclcMultimodalClsModel

import torch.backends.cuda as cuda_bk
import torch.backends.cudnn as cudnn_bk

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_train_val_indices(n: int, val_frac: float, seed: int = 42):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_train = int(n * (1.0 - val_frac))
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:].tolist()
    return train_idx, val_idx


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    scaler: GradScaler | None = None,
):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        image = batch["image"].to(device)  # [B, 2, D, H, W]
        ehr = batch["ehr"].to(device)  # [B, F]
        label = batch["label"].to(device).float()  # [B]

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            # plain FP32
            logits, _ = model(image, ehr)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
        else:
            # AMP
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits, _ = model(image, ehr)
                loss = loss_fn(logits, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs = image.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    amp: bool = False,
    return_preds: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probs = []

    for batch in loader:
        image = batch["image"].to(device)
        ehr = batch["ehr"].to(device)
        label = batch["label"].to(device).float()

        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits, _ = model(image, ehr)
                loss = loss_fn(logits, label)
        else:
            logits, _ = model(image, ehr)
            loss = loss_fn(logits, label)

        bs = image.size(0)
        total_loss += float(loss.item()) * bs
        total_samples += bs

        all_labels.append(label.cpu().numpy())
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())

    if total_samples == 0:
        if return_preds:
            return float("nan"), float("nan"), float("nan"), np.array([]), np.array([])
        return float("nan"), float("nan"), float("nan")

    mean_loss = total_loss / total_samples
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    # Edge cases where all labels are same
    if np.unique(labels).size < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))

    if return_preds:
        return mean_loss, auroc, auprc, labels, probs
    return mean_loss, auroc, auprc


def main():
    parser = argparse.ArgumentParser(
        description="Train NSCLC multimodal classifier (image + EHR) for 3-year OS."
    )
    parser.add_argument(
        "--manifest-csv",
        required=True,
        help="Path to nsclc_manifest.csv with case_id, pet_path, ct_path, event_time, event_indicator",
    )
    parser.add_argument(
        "--ehr-csv",
        required=True,
        help="Path to NSCLC_EHR_features.csv with patient_id and features",
    )
    parser.add_argument(
        "--cutoff-days",
        type=float,
        default=1096.0,
        help="Cutoff in days for early-death label (default 1096 ~ 3 years)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device, e.g. 'cuda' or 'cuda:0' or 'cpu'",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/azureuser/PITL/outputs/multimodal_cls",
        help="Directory to save best model and logs",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=15,
        help="Stop if val AUROC does not improve for this many epochs",
    )
    parser.add_argument(
        "--pet-clip",
        type=float,
        default=35.0,
        help="PET SUV clip used in get_nsclc_transforms",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[4.0, 4.0, 4.0],
        help="Target spacing in mm, e.g. --spacing 4 4 4",
    )
    parser.add_argument(
        "--patch",
        type=int,
        default=None,
        help="Optional patch size, e.g. 96 for 96x96x96 crops",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (autocast + GradScaler)",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if device.type == "cuda":
        cuda_bk.matmul.allow_tf32 = True
        cudnn_bk.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Loading multimodal dataset")

    # Base dataset just to inspect dimension and length
    base_ds = NsclcMultimodalDataset(
        manifest_csv=args.manifest_csv,
        ehr_csv=args.ehr_csv,
        transforms=None,
        cutoff_days=args.cutoff_days,
    )
    n_total = len(base_ds)
    ehr_input_dim = len(base_ds.ehr_cols)
    print(f"[info] Total patients: {n_total}")
    print(f"[info] EHR feature dim: {ehr_input_dim}")

    # Build train and val transforms
    train_tf = get_nsclc_transforms(
        train=True,
        pet_key="pet",
        ct_key="ct",
        pet_clip=args.pet_clip,
        patch=args.patch,
        spacing=tuple(args.spacing),
    )
    val_tf = get_nsclc_transforms(
        train=False,
        pet_key="pet",
        ct_key="ct",
        pet_clip=args.pet_clip,
        patch=args.patch,
        spacing=tuple(args.spacing),
    )

    # Two copies of the dataset with different transforms
    ds_train_full = NsclcMultimodalDataset(
        manifest_csv=args.manifest_csv,
        ehr_csv=args.ehr_csv,
        transforms=train_tf,
        cutoff_days=args.cutoff_days,
    )
    ds_val_full = NsclcMultimodalDataset(
        manifest_csv=args.manifest_csv,
        ehr_csv=args.ehr_csv,
        transforms=val_tf,
        cutoff_days=args.cutoff_days,
    )

    # Sanity: lengths must match
    assert len(ds_train_full) == n_total == len(ds_val_full)

    # Train/val indices
    train_idx, val_idx = make_train_val_indices(
        n_total, val_frac=args.val_frac, seed=args.seed
    )
    print(f"[info] Train size: {len(train_idx)}, Val size: {len(val_idx)}")

    train_ds = Subset(ds_train_full, train_idx)
    val_ds = Subset(ds_val_full, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_list_data_collate,
    )

    # Model
    model = NsclcMultimodalClsModel(
        ehr_input_dim=ehr_input_dim, img_emb_dim=512, ehr_emb_dim=128
    )

    # simple multi GPU if available and using cuda
    if (
        torch.cuda.is_available()
        and args.device == "cuda"
        and torch.cuda.device_count() > 1
    ):
        print(f"[info] Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    scaler = GradScaler(enabled=args.amp)

    best_auroc = -np.inf
    best_state = None
    epochs_no_improve = 0

    epochs_log = []
    train_loss_log = []
    val_loss_log = []
    val_auroc_log = []
    val_auprc_log = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, scaler=scaler
        )
        val_loss, val_auroc, val_auprc = eval_epoch(
            model, val_loader, device, loss_fn, amp=args.amp
        )
        scheduler.step()

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_AUROC={val_auroc:.4f}  "
            f"val_AUPRC={val_auprc:.4f}"
        )

        epochs_log.append(epoch)
        train_loss_log.append(train_loss)
        val_loss_log.append(val_loss)
        val_auroc_log.append(val_auroc)
        val_auprc_log.append(val_auprc)

        improved = np.isfinite(val_auroc) and (val_auroc > best_auroc + 1e-4)
        if improved:
            best_auroc = val_auroc
            best_state = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "best_val_auroc": best_auroc,
                    "ehr_input_dim": ehr_input_dim,
                    "cutoff_days": args.cutoff_days,
                    "manifest_csv": args.manifest_csv,
                    "ehr_csv": args.ehr_csv,
                },
                out_dir / "best_multimodal_cls.pt",
            )
            print(
                f"[info] New best model saved with val_AUROC={best_auroc:.4f} "
                f"at epoch {epoch}"
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(
                    f"[info] Early stopping at epoch {epoch} "
                    f"(no val AUROC improvement for {epochs_no_improve} epochs)"
                )
                break

    log_df = pd.DataFrame(
        {
            "epoch": epochs_log,
            "train_loss": train_loss_log,
            "val_loss": val_loss_log,
            "val_AUROC": val_auroc_log,
            "val_AUPRC": val_auprc_log,
        }
    )
    log_df.to_csv(out_dir / "metrics_multimodal_cls.csv", index=False)

    if len(epochs_log) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        ax0 = axes[0]
        ax0.plot(epochs_log, train_loss_log, label="train_loss")
        ax0.plot(epochs_log, val_loss_log, label="val_loss")
        ax0.set_xlabel("epoch")
        ax0.set_ylabel("BCE loss")
        ax0.set_title("Train vs Val loss")
        ax0.legend()

        ax1 = axes[1]
        ax1.plot(epochs_log, val_auroc_log, label="val_AUROC")
        ax1.plot(epochs_log, val_auprc_log, label="val_AUPRC")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("metric")
        ax1.set_title("Val AUROC / AUPRC")
        ax1.legend()

        fig.tight_layout()
        plt.savefig(out_dir / "curves_multimodal_cls.png", dpi=150)
        plt.close(fig)

        if best_state is not None:
            model.load_state_dict(best_state)
            print(
                f"[info] Training done. Best val_AUROC={best_auroc:.4f}. "
                f"Model weights are in {out_dir / 'best_multimodal_cls.pt'}"
            )

            # recompute metrics for best checkpoint and full val set
            val_loss_best, val_auroc_best, val_auprc_best, y_true, y_prob = eval_epoch(
                model,
                val_loader,
                device,
                loss_fn,
                amp=args.amp,
                return_preds=True,
            )

            if y_true.size > 0:
                y_true = y_true.astype(int)
                y_pred = (y_prob >= 0.5).astype(int)

                acc = accuracy_score(y_true, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="binary", zero_division=0
                )

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                pos_frac = y_true.mean()
                print("[info] Detailed validation metrics at best checkpoint:")
                print(
                    f"       loss={val_loss_best:.4f}, "
                    f"AUROC={val_auroc_best:.4f}, AUPRC={val_auprc_best:.4f}"
                )
                print(
                    f"       Accuracy={acc:.4f}, "
                    f"Precision={precision:.4f}, "
                    f"Recall={recall:.4f}, "
                    f"F1={f1:.4f}"
                )
                print(
                    f"       Confusion matrix [[TN, FP], [FN, TP]] = "
                    f"[[{tn}, {fp}], [{fn}, {tp}]]"
                )
                print(
                    f"       Positives in val set: {int(y_true.sum())} / {len(y_true)} "
                    f"({pos_frac:.3f} fraction)"
                )

                # save to file
                with open(out_dir / "val_metrics_best.txt", "w") as f:
                    f.write(
                        f"loss={val_loss_best:.4f}\n"
                        f"AUROC={val_auroc_best:.4f}\n"
                        f"AUPRC={val_auprc_best:.4f}\n"
                        f"Accuracy={acc:.4f}\n"
                        f"Precision={precision:.4f}\n"
                        f"Recall={recall:.4f}\n"
                        f"F1={f1:.4f}\n"
                        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n"
                        f"positives={int(y_true.sum())} / {len(y_true)} "
                        f"({pos_frac:.3f})\n"
                    )
            else:
                print("[warn] No validation samples found for final metrics.")
        else:
            print("[warn] No valid improvement found. Model was not saved.")


if __name__ == "__main__":
    main()
