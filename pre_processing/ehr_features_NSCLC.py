"""
Example usage:

python ehr_features_NSCLC.py \
  --csv "/data/tcia_out/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv" \
  --out-csv "/home/azureuser/PITL/pre_processing/NSCLC_EHR_features.csv" \
  --stats-json "/home/azureuser/PITL/pre_processing/NSCLC_EHR_stats.json" \
  --id-col "Case ID" \
  --pairing-log "/home/azureuser/PITL/audit_and_convert/logs_after_conversion/pairing_log.csv"
"""

import argparse, json, re
import pandas as pd, numpy as np

# Default binary columns (tri-state: value + unknown flag)
DEFAULT_BINARY_COLS = [
    "EGFR mutation status",
    "KRAS mutation status",
    "ALK translocation status",
    "Adjuvant Treatment",
    "Chemotherapy",
    "Radiation",
]

# Single continuous field for now
DEFAULT_CONTINUOUS_COL = "Days between CT and surgery"

# Outcome / leak columns (never used as inputs)
LEAK_COLUMNS = set(
    [
        "Survival Status",
        "Date of Death",
        "Time to Death (days)",
        "Date of Last Known Alive",
        "Recurrence",
        "Recurrence Location",
        "Date of Recurrence",
    ]
)

# Tokens indicating positives / negatives
POS_TOKENS = {
    "yes",
    "y",
    "true",
    "1",
    "positive",
    "pos",
    "mut",
    "mutant",
    "mutated",
    "present",
    "received",
    "done",
    "given",
    "administered",
    "translocated",
}

NEG_TOKENS = {
    "no",
    "n",
    "false",
    "0",
    "negative",
    "neg",
    "wildtype",
    "wt",
    "absent",
    "not received",
    "none",
}

# Explicit unknown tokens from your CSV
UNK_TOKENS = {
    "unknown",
    "not collected",
    "not assessed",
    "",  # empty after normalization
}


def _norm_str(x):
    """Lowercase, strip, collapse whitespace."""
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_binary(series: pd.Series):
    """
    Map arbitrary text to (bin, unknown).

    - bin in {0,1}
    - unknown in {0,1}

    Rules tuned for your NSCLC CSV:
    - 'Mutant' / 'Translocated' / 'Yes' / 'Present' -> (1, 0)
    - 'Wildtype' / 'No' / 'Absent' -> (0, 0)
    - 'Unknown', 'Not collected', 'Not Assessed', empty -> (0, 1)
    Anything else falls back to unknown.
    """
    vals = series.apply(_norm_str)
    binv, unk = [], []
    for v in vals:
        # 1) Explicit unknowns first
        if v in UNK_TOKENS:
            binv.append(0)
            unk.append(1)
            continue

        # 2) Explicit positives
        if v in POS_TOKENS or any(
            tok in v
            for tok in [
                "mutant",
                "mutated",
                "positive",
                "received",
                "done",
                "given",
                "administered",
                "translocated",
                "present",
            ]
        ):
            binv.append(1)
            unk.append(0)
            continue

        # 3) Explicit negatives
        if v in NEG_TOKENS or any(
            tok in v
            for tok in ["wildtype", "wt", "negative", "no", "absent", "not received"]
        ):
            binv.append(0)
            unk.append(0)
            continue

        # 4) Conservative fallback: treat ambiguous as unknown
        binv.append(0)
        unk.append(1)

    return pd.Series(binv, index=series.index, dtype=int), pd.Series(
        unk, index=series.index, dtype=int
    )


def zscore(series: pd.Series):
    x = pd.to_numeric(series, errors="coerce")
    miss = x.isna().astype(int)
    mu = float(np.nanmean(x.values)) if np.any(~np.isnan(x.values)) else 0.0
    sd = float(np.nanstd(x.values)) if np.any(~np.isnan(x.values)) else 1.0
    if sd == 0.0 or not np.isfinite(sd):
        sd = 1.0
    xz = (x.fillna(mu) - mu) / sd
    return xz.astype(float), miss, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--stats-json", required=True)
    ap.add_argument("--id-col", default="Case ID")
    ap.add_argument("--binary-cols", default=",".join(DEFAULT_BINARY_COLS))
    ap.add_argument("--continuous-col", default=DEFAULT_CONTINUOUS_COL)
    ap.add_argument(
        "--pairing-log",
        default=None,
        help="Optional path to pairing_log.csv. If given, only patients whose IDs "
        "appear in pairing_log['patient_id'] are kept.",
    )
    args = ap.parse_args()

    # Read main NSCLC CSV
    df = pd.read_csv(args.csv)
    id_col = args.id_col
    if id_col not in df.columns:
        raise SystemExit(
            f"ID column '{id_col}' not found. Available: {list(df.columns)[:10]}..."
        )

    # Optional filtering by pairing_log.csv
    if args.pairing_log is not None:
        pair_df = pd.read_csv(args.pairing_log)
        if "patient_id" not in pair_df.columns:
            raise SystemExit(
                f"'patient_id' column not found in pairing log {args.pairing_log}. "
                f"Available columns: {list(pair_df.columns)}"
            )
        keep_ids = set(pair_df["patient_id"].astype(str))
        df[id_col] = df[id_col].astype(str)
        before = len(df)
        df = df[df[id_col].isin(keep_ids)].copy()
        after = len(df)
        print(
            f"Filtered NSCLC CSV using pairing_log: {before} -> {after} rows "
            f"matching {len(keep_ids)} patient_id values."
        )

    bin_cols = [c.strip() for c in args.binary_cols.split(",") if c.strip()]
    cont_col = args.continuous_col if args.continuous_col in df.columns else None

    # Build features only for the filtered set
    feats = pd.DataFrame({"patient_id": df[id_col]})
    schema = {
        "id_col": id_col,
        "binary_cols": [],
        "continuous_col": cont_col,
        "zscore_mu_sd": {},
        "dropped_leak_columns": sorted(
            list(c for c in df.columns if c in LEAK_COLUMNS)
        ),
        "unknown_tokens": sorted(list(UNK_TOKENS)),
        "filtered_by_pairing_log": args.pairing_log is not None,
    }

    # Binary features
    for col in bin_cols:
        if col not in df.columns:
            print(f"[WARN] Binary column '{col}' not found in CSV. Skipping.")
            continue
        binv, unk = parse_binary(df[col])
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", col.strip())
        feats[f"{safe}_bin"] = binv
        feats[f"{safe}_unknown"] = unk
        schema["binary_cols"].append(col)

    # Continuous feature (Days between CT and surgery) with z score + missing flag
    if cont_col is not None:
        xz, miss, mu, sd = zscore(df[cont_col])
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", cont_col.strip())
        feats[f"{safe}_z"] = xz
        feats[f"{safe}_missing"] = miss
        schema["zscore_mu_sd"][cont_col] = {"mu": mu, "sd": sd}

    feats.to_csv(args.out_csv, index=False)
    with open(args.stats_json, "w") as f:
        json.dump(schema, f, indent=2)
    print("Wrote features to", args.out_csv)
    print("Wrote stats to", args.stats_json)
    print("Feature dim:", feats.drop(columns=["patient_id"]).shape[1])


if __name__ == "__main__":
    main()
