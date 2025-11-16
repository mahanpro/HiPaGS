"""
Build NSCLC manifest for image encoder training.

Inputs:
  1) labeling.csv         (OS labels)
       columns: patient_id, event_time_days, event_indicator, cls_label, ...

  2) conversion_log.csv   (PET/CT NIfTI paths for kept pairs)
       columns: patient_id, pet_path, ct_path, ...

Output:
  nsclc_manifest.csv with columns:
    case_id, pet_path, ct_path, event_time, event_indicator, [optional cls_label]

Example usage:

python create_manifest.py \
  --labeling "/home/azureuser/PITL/pre_processing/labeling.csv" \
  --conversion-log "/home/azureuser/PITL/audit_and_convert/logs_after_conversion/conversion_log.csv" \
  --out-csv "/home/azureuser/PITL/pre_processing/nsclc_manifest.csv"
"""

import argparse
import pandas as pd


def build_manifest(
    labeling_csv: str,
    conversion_log_csv: str,
    out_csv: str,
    label_id_col: str = "patient_id",
    event_time_col: str = "event_time_days",
    event_indicator_col: str = "event_indicator",
    conv_id_col: str = "patient_id",
    cls_label_col: str = "cls_label",
):
    print(f"[info] Reading labeling from: {labeling_csv}")
    labeling = pd.read_csv(labeling_csv)

    print(f"[info] Reading conversion log from: {conversion_log_csv}")
    conv = pd.read_csv(conversion_log_csv)

    # Optional: if a class_keep flag exists, restrict to kept pairs only
    if "class_keep" in conv.columns:
        before = len(conv)
        conv = conv[conv["class_keep"] == "yes"].copy()
        after = len(conv)
        print(
            f"[info] Filtered conversion_log by class_keep == 'yes': {before} -> {after}"
        )

    # Basic column checks
    required_label_cols = {label_id_col, event_time_col, event_indicator_col}
    missing_label = required_label_cols.difference(labeling.columns)
    if missing_label:
        raise ValueError(f"Missing columns in labeling.csv: {missing_label}")

    required_conv_cols = {conv_id_col, "pet_path", "ct_path"}
    missing_conv = required_conv_cols.difference(conv.columns)
    if missing_conv:
        raise ValueError(f"Missing columns in conversion_log.csv: {missing_conv}")

    # Restrict conversion log to needed columns
    conv_small = conv[[conv_id_col, "pet_path", "ct_path"]].copy()

    # Report overlaps
    label_ids = set(labeling[label_id_col].astype(str))
    conv_ids = set(conv_small[conv_id_col].astype(str))

    common_ids = label_ids & conv_ids
    label_only = sorted(label_ids - conv_ids)
    conv_only = sorted(conv_ids - label_ids)

    print(f"[info] # patients in labeling:        {len(label_ids)}")
    print(f"[info] # patients in conversion_log:  {len(conv_ids)}")
    print(f"[info] # patients in intersection:    {len(common_ids)}")

    if label_only:
        print(f"[warn] Patients in labeling only (no PET/CT pair): {label_only}")
    if conv_only:
        print(f"[warn] Patients in conversion_log only (no OS label): {conv_only}")

    # Inner join to keep only patients with both labels and PET/CT
    merged = labeling.merge(
        conv_small,
        left_on=label_id_col,
        right_on=conv_id_col,
        how="inner",
        validate="one_to_one",
    )

    # Drop any rows with missing PET or CT paths just in case
    merged = merged.dropna(subset=["pet_path", "ct_path"])

    # Rename to the requested schema
    rename_map = {
        label_id_col: "case_id",
        event_time_col: "event_time",
        event_indicator_col: "event_indicator",
    }
    if cls_label_col in merged.columns:
        rename_map[cls_label_col] = "cls_label"

    manifest = merged.rename(columns=rename_map)

    # Select and order columns
    cols = ["case_id", "pet_path", "ct_path", "event_time", "event_indicator"]
    if "cls_label" in manifest.columns:
        cols.append("cls_label")

    manifest = manifest[cols]

    print(f"[info] Final manifest rows: {len(manifest)}")

    # Save
    manifest.to_csv(out_csv, index=False)
    print(f"[info] Wrote manifest to: {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Build NSCLC manifest CSV for image encoder training."
    )
    p.add_argument(
        "--labeling",
        dest="labeling_csv",
        required=True,
        help="Path to labeling.csv with OS labels",
    )
    p.add_argument(
        "--conversion-log",
        dest="conversion_log_csv",
        required=True,
        help="Path to conversion_log.csv (or pairing_log.csv) with PET/CT paths",
    )
    p.add_argument(
        "--out-csv",
        dest="out_csv",
        required=True,
        help="Output path for nsclc_manifest.csv",
    )

    # Optional overrides if your column names ever change
    p.add_argument(
        "--label-id-col",
        default="patient_id",
        help="ID column in labeling.csv (default: patient_id)",
    )
    p.add_argument(
        "--event-time-col",
        default="event_time_days",
        help="Event time column in labeling.csv (default: event_time_days)",
    )
    p.add_argument(
        "--event-indicator-col",
        default="event_indicator",
        help="Event indicator column in labeling.csv (default: event_indicator)",
    )
    p.add_argument(
        "--conv-id-col",
        default="patient_id",
        help="ID column in conversion_log.csv (default: patient_id)",
    )
    p.add_argument(
        "--cls-label-col",
        default="cls_label",
        help="Classification label column in labeling.csv (default: cls_label)",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_manifest(
        labeling_csv=args.labeling_csv,
        conversion_log_csv=args.conversion_log_csv,
        out_csv=args.out_csv,
        label_id_col=args.label_id_col,
        event_time_col=args.event_time_col,
        event_indicator_col=args.event_indicator_col,
        conv_id_col=args.conv_id_col,
        cls_label_col=args.cls_label_col,
    )
