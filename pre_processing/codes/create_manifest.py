"""
Melanoma:

python /home/azureuser/PITL/pre_processing/codes/create_manifest.py \
  --labeling /home/azureuser/PITL/pre_processing/data/Melanoma/Melanoma_labels.csv \
  --conversion /home/azureuser/PITL/pre_processing/data/Melanoma/Melanoma_paths.csv \
  --dataset-name Melanoma \
  --out-csv /home/azureuser/PITL/pre_processing/data/Melanoma/Melanoma_manifest.csv
  
NSCLC:

python /home/azureuser/PITL/pre_processing/codes/create_manifest.py \
  --labeling /home/azureuser/PITL/pre_processing/data/NSCLC/NSCLC_labels.csv \
  --conversion /home/azureuser/PITL/audit_and_convert/logs_after_conversion/NSCLC/conversion_log.csv \
  --dataset-name NSCLC \
  --out-csv /home/azureuser/PITL/pre_processing/data/NSCLC/NSCLC_manifest.csv

LUSC:

python /home/azureuser/PITL/pre_processing/codes/create_manifest.py \
  --labeling /home/azureuser/PITL/pre_processing/data/TCGA/LUSC/TCGA_LUSC_labels.csv \
  --conversion /home/azureuser/PITL/audit_and_convert/logs_after_conversion/TCGA/LUSC/conversion_log.csv \
  --dataset-name TCGA_LUSC \
  --out-csv /home/azureuser/PITL/pre_processing/data/TCGA/LUSC/TCGA_LUSC_manifest.csv

"""

import argparse
import pandas as pd


def build_manifest_general(
    labeling_csv: str,
    conversion_csv: str,
    out_csv: str,
    label_id_col: str = "patient_id",
    conv_id_col: str = "patient_id",
    dataset_name: str | None = None,
    require_any_image: bool = True,
):
    """
    General manifest builder for multi-cohort training.

    Assumptions:
      - labeling_csv has columns:
          label_id_col, event_time_days, event_indicator, cls_label (optional)
      - conversion_csv has at least:
          conv_id_col, and one or more of:
            ct_path, pet_path, petct_path
        or whatever you choose to name the path columns.

    The output manifest has:
      dataset, patient_id, event_time_days, event_indicator, cls_label,
      ct_path, pet_path, petct_path

    Rows are kept if:
      - labels exist, and
      - (require_any_image = False) or (at least one of the path columns is non-null).
    """

    print(f"[info] Reading labels from: {labeling_csv}")
    labels = pd.read_csv(labeling_csv)

    print(f"[info] Reading conversion from: {conversion_csv}")
    conv = pd.read_csv(conversion_csv)

    # Basic column checks for labels
    for col in [label_id_col, "event_time_days", "event_indicator"]:
        if col not in labels.columns:
            raise ValueError(f"Missing '{col}' in labeling_csv")

    # Make sure ID columns are strings
    labels[label_id_col] = labels[label_id_col].astype(str)
    conv[conv_id_col] = conv[conv_id_col].astype(str)

    # Identify possible path columns in conversion
    path_cols = [c for c in conv.columns if c.endswith("_path")]
    if not path_cols:
        raise ValueError(
            f"No *_path columns found in conversion_csv. "
            f"Columns are: {list(conv.columns)}"
        )

    # Collapse conversion to one row per patient_id by simple first occurrence rule
    # (you can replace with more complex logic if needed)
    conv_sorted = conv.sort_values(by=[conv_id_col])
    conv_unique = conv_sorted.drop_duplicates(subset=[conv_id_col], keep="first")

    merged = labels.merge(
        conv_unique,
        left_on=label_id_col,
        right_on=conv_id_col,
        how="left",
        validate="one_to_one",
    )

    # If you want to require at least one path
    if require_any_image:
        non_null_any = merged[path_cols].notna().any(axis=1)
        before = len(merged)
        merged = merged[non_null_any].copy()
        after = len(merged)
        print(f"[info] Dropped {before - after} rows with no image paths")

    # Rename columns
    rename_map = {
        label_id_col: "patient_id",
        "event_time_days": "event_time_days",
        "event_indicator": "event_indicator",
    }
    if "cls_label" in merged.columns:
        rename_map["cls_label"] = "cls_label"

    manifest = merged.rename(columns=rename_map)

    # Add dataset column
    if dataset_name is not None:
        manifest["dataset"] = dataset_name
    else:
        manifest["dataset"] = "unknown"

    # Ensure all expected path columns exist in output
    for col in ["ct_path", "pet_path", "petct_path"]:
        if col not in manifest.columns:
            manifest[col] = pd.NA

    keep_cols = [
        "dataset",
        "patient_id",
        "event_time_days",
        "event_indicator",
        "cls_label",
        "ct_path",
        "pet_path",
        "petct_path",
    ]
    keep_cols = [c for c in keep_cols if c in manifest.columns]
    manifest = manifest[keep_cols]

    print(f"[info] Final manifest rows: {len(manifest)}")
    manifest.to_csv(out_csv, index=False)
    print(f"[info] Wrote manifest to: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(
        description="General manifest builder for multi-cohort survival classification",
    )
    p.add_argument("--labeling", required=True, help="Labels CSV path")
    p.add_argument("--conversion", required=True, help="Conversion CSV path")
    p.add_argument("--out-csv", required=True, help="Output manifest CSV")
    p.add_argument("--label-id-col", default="patient_id")
    p.add_argument("--conv-id-col", default="patient_id")
    p.add_argument("--dataset-name", default=None)
    p.add_argument(
        "--no-require-any-image",
        action="store_true",
        help="If set, do not drop rows with no image paths",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_manifest_general(
        labeling_csv=args.labeling,
        conversion_csv=args.conversion,
        out_csv=args.out_csv,
        label_id_col=args.label_id_col,
        conv_id_col=args.conv_id_col,
        dataset_name=args.dataset_name,
        require_any_image=not args.no_require_any_image,
    )
