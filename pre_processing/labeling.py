#!/usr/bin/env python3
"""
NSCLC OS Label Builder (+ leakage checks)

Usage:
  python labeling.py \
    --csv "/data/tcia_out/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv" \
    --out-csv "/home/azureuser/PITL/pre_processing/labeling.csv" \
    --id-col "Case ID" \
    --ct-col "CT Date" \
    --pet-col "PET Date" \
    --dod-col "Date of Death" \
    --dla-col "Date of Last Known Alive" \
    --status-col "Survival Status" \
    --pairing-log "/home/azureuser/PITL/audit_and_convert/logs_after_conversion/pairing_log.csv"

What it does
- Optionally filters to patients present in pairing_log['patient_id'] if --pairing-log is given.
- Chooses baseline = earliest of PET Date and CT Date available per patient.
- Computes event time:
    If Date of Death present: days from baseline to Date of Death.
    Else: days from baseline to Date of Last Known Alive (censor).
- Computes event indicator:
    1 if death date present OR status indicates deceased; 0 otherwise.
- Performs acceptance checks:
    * Flags negative or zero durations.
    * Reports censoring fraction.
    * Reports N, events, median follow-up (reverse KM), and median OS (KM, if reached).

Output CSV columns (per patient)
- patient_id
- baseline_date
- baseline_source  (one of: "PET", "CT", "PET|CT", "MISSING")
- event_time_days
- event_indicator  (1=death observed, 0=censored)
- notes            (warnings like NEGATIVE_OR_ZERO_TIME or BASELINE_MISSING)

Leakage note: Do NOT feed these columns as model inputs:
- Survival Status, Date of Death, Time to Death, Date of Last Known Alive,
  Recurrence, Recurrence Location, Date of Recurrence.
"""
import argparse
import pandas as pd
import numpy as np


def _to_date(series):
    # Robust datetime parser: returns pandas datetime (UTC-naive)
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)


def _norm_status(x):
    # Normalize survival status to lowercase alphanum for mapping
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace())


DEAD_TOKENS = {"dead", "deceased", "expired", "1", "true", "yes", "death", "died"}
ALIVE_TOKENS = {"alive", "living", "0", "false", "no", "censored"}


def km_median_time(durations, events):
    """
    Kaplan-Meier median event time.
    durations: array-like of positive times
    events: boolean array, True for event observed, False for censored
    Returns (median_time_or_nan, reached_bool)
    """
    arr = np.asarray(durations, dtype=float)
    ev = np.asarray(events, dtype=bool)
    valid = np.isfinite(arr) & (arr > 0)
    arr = arr[valid]
    ev = ev[valid]
    if arr.size == 0:
        return (np.nan, False)
    order = np.argsort(arr, kind="mergesort")
    t = arr[order]
    d = ev[order].astype(int)
    # Unique times
    uniq_t, idx = np.unique(t, return_index=True)
    N = t.size
    n_at_risk = np.empty_like(uniq_t, dtype=float)
    d_at = np.empty_like(uniq_t, dtype=float)
    for k, s in enumerate(idx):
        e = idx[k + 1] if k + 1 < idx.size else N
        d_at[k] = d[s:e].sum()
        n_at_risk[k] = N - s
    # KM survival
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = 1.0 - (d_at / n_at_risk)
    frac[np.isnan(frac)] = 1.0
    S = np.cumprod(frac)
    hit = np.where(S <= 0.5)[0]
    if hit.size == 0:
        return (np.nan, False)
    return (float(uniq_t[hit[0]]), True)


def reverse_km_median_followup(durations, events):
    """
    Reverse KM to estimate median follow-up time (censoring distribution).
    Treat censored as events, events as censored.
    """
    return km_median_time(durations, ~np.asarray(events, dtype=bool))


def build_os_labels(
    df: pd.DataFrame,
    id_col: str,
    ct_col: str,
    pet_col: str,
    dod_col: str,
    dla_col: str,
    status_col: str,
):
    df = df.copy()

    # Parse dates
    ct = _to_date(df.get(ct_col))
    pet = _to_date(df.get(pet_col))
    dod = _to_date(df.get(dod_col))
    dla = _to_date(df.get(dla_col))

    # Baseline = earliest of PET and CT that exist
    baseline = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    baseline_source = pd.Series("MISSING", index=df.index, dtype=object)

    both = (~pet.isna()) & (~ct.isna())
    only_pet = (~pet.isna()) & (ct.isna())
    only_ct = (pet.isna()) & (~ct.isna())

    baseline[both] = pet[both].where(pet[both] <= ct[both], ct[both])
    baseline_source[both] = "PET|CT"
    baseline[only_pet] = pet[only_pet]
    baseline_source[only_pet] = "PET"
    baseline[only_ct] = ct[only_ct]
    baseline_source[only_ct] = "CT"

    # Event indicator: 1 if death date present OR status in DEAD_TOKENS, else 0
    status_norm = (
        df.get(status_col).map(_norm_status)
        if status_col in df.columns
        else pd.Series("", index=df.index)
    )
    status_dead = status_norm.isin(DEAD_TOKENS)
    event_indicator = ((~dod.isna()) | status_dead).astype(int)

    # Event time: to DoD if present; else to DLA
    end_date = dod.fillna(dla)
    event_time_days = (end_date - baseline).dt.days

    # Notes and checks
    notes = pd.Series("", index=df.index, dtype=object)
    neg_or_zero = event_time_days <= 0
    baseline_missing = baseline.isna()
    end_missing = end_date.isna()

    def _append_flag(mask, flag):
        idx = mask.fillna(False)
        notes.loc[idx] = (notes.loc[idx].astype(str) + ";" + flag).str.strip(";")

    _append_flag(neg_or_zero, "NEGATIVE_OR_ZERO_TIME")
    _append_flag(baseline_missing, "BASELINE_MISSING")
    _append_flag(end_missing, "ENDDATE_MISSING")

    out = pd.DataFrame(
        {
            "patient_id": df[id_col] if id_col in df.columns else np.arange(len(df)),
            "baseline_date": baseline,
            "baseline_source": baseline_source,
            "event_time_days": event_time_days,
            "event_indicator": event_indicator,
            "notes": notes.replace("", np.nan),
        }
    )

    # Acceptance summary
    N = len(out)
    events = int(out["event_indicator"].sum())
    censored = N - events
    censor_frac = censored / N if N > 0 else np.nan

    med_os, os_reached = km_median_time(
        out["event_time_days"].values.astype(float),
        out["event_indicator"].values.astype(bool),
    )
    med_fu, fu_reached = reverse_km_median_followup(
        out["event_time_days"].values.astype(float),
        out["event_indicator"].values.astype(bool),
    )

    summary = {
        "N": N,
        "events": events,
        "censored": censored,
        "censoring_fraction": (
            round(censor_frac, 4) if np.isfinite(censor_frac) else np.nan
        ),
        "median_OS_days": (
            None if (not os_reached or not np.isfinite(med_os)) else float(med_os)
        ),
        "median_followup_days": (
            None if (not fu_reached or not np.isfinite(med_fu)) else float(med_fu)
        ),
        "num_negative_or_zero_times": int(
            np.nan_to_num(neg_or_zero).sum() if hasattr(neg_or_zero, "sum") else 0
        ),
        "num_baseline_missing": int(
            np.nan_to_num(baseline_missing).sum()
            if hasattr(baseline_missing, "sum")
            else 0
        ),
        "num_enddate_missing": int(
            np.nan_to_num(end_missing).sum() if hasattr(end_missing, "sum") else 0
        ),
    }

    return out, summary


def main():
    parser = argparse.ArgumentParser(
        description="Build Overall Survival labels for NSCLC Radiogenomics."
    )
    parser.add_argument(
        "--csv", required=True, help="Path to input CSV containing NSCLC fields"
    )
    parser.add_argument("--out-csv", required=True, help="Path to write labels CSV")
    parser.add_argument(
        "--id-col", default="Patient ID", help="Patient identifier column name"
    )
    parser.add_argument("--ct-col", default="CT Date", help="CT date column name")
    parser.add_argument("--pet-col", default="PET Date", help="PET date column name")
    parser.add_argument(
        "--dod-col", default="Date of Death", help="Date of death column name"
    )
    parser.add_argument(
        "--dla-col",
        default="Date of Last Known Alive",
        help="Date of last known alive column name",
    )
    parser.add_argument(
        "--status-col", default="Survival Status", help="Survival status column name"
    )
    parser.add_argument(
        "--pairing-log",
        default=None,
        help=(
            "Optional path to pairing_log.csv. If given, only patients whose IDs "
            "appear in pairing_log['patient_id'] are kept."
        ),
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # Optional filtering by pairing_log.csv, just like ehr_features_NSCLC.py
    if args.pairing_log is not None:
        pair_df = pd.read_csv(args.pairing_log)
        if "patient_id" not in pair_df.columns:
            raise SystemExit(
                f"'patient_id' column not found in pairing log {args.pairing_log}. "
                f"Available columns: {list(pair_df.columns)}"
            )
        keep_ids = set(pair_df["patient_id"].astype(str))
        if args.id_col not in df.columns:
            raise SystemExit(
                f"ID column '{args.id_col}' not found in CSV. "
                f"Available: {list(df.columns)[:10]}..."
            )
        df[args.id_col] = df[args.id_col].astype(str)
        before = len(df)
        df = df[df[args.id_col].isin(keep_ids)].copy()
        after = len(df)
        print(
            f"Filtered input CSV using pairing_log: {before} -> {after} rows "
            f"matching {len(keep_ids)} patient_id values."
        )

    labels, summary = build_os_labels(
        df,
        id_col=args.id_col,
        ct_col=args.ct_col,
        pet_col=args.pet_col,
        dod_col=args.dod_col,
        dla_col=args.dla_col,
        status_col=args.status_col,
    )

    labels.to_csv(args.out_csv, index=False)

    print("=== OS Label Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nWrote labels to:", args.out_csv)


if __name__ == "__main__":
    main()
