#!/usr/bin/env python3
"""
Unified OS label builder for NSCLC, TCGA (LUAD, LUSC, BLCA, LIHC, UCEC),
and the private Melanoma PET/CT cohort.

It does three things:

  - NSCLC:
      Reproduce labeling.py behavior
      Baseline = earliest of CT Date and PET Date
      Event time = (Date of Death or Last Known Alive) - baseline

  - TCGA (LUAD, LUSC, BLCA, LIHC, UCEC):
      Use TCGA-CDR-SupplementalTableS1.xlsx (OS.time, vital_status)
      Baseline is implicitly diagnosis in OS.time
      Event time = OS.time

  - Melanoma:
      Use Cases_PETCT_staging_Melanoma*.xlsx
      Baseline = date_staging_pet (staging PET/CT)
      Event time = date_of_death_last_fu - date_staging_pet
      Event indicator = 1 if event == 'dead', else 0
      This gives OS from staging PET/CT.

All branches also build a binary classification label (cls_label) for a
user defined horizon in years (cls_cutoff_years).

Usage examples
--------------

# NSCLC (restricted to imaging patients)
python /home/azureuser/PITL/pre_processing/codes/build_os_labels_multi.py \
  --dataset nsclc \
  --nsclc-csv "/data/tcia_out/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv" \
  --out-csv "/home/azureuser/PITL/pre_processing/data/NSCLC/NSCLC_labels.csv" \
  --id-col "Case ID" \
  --ct-col "CT Date" \
  --pet-col "PET Date" \
  --dod-col "Date of Death" \
  --dla-col "Date of Last Known Alive" \
  --status-col "Survival Status" \
  --conversion-log "/home/azureuser/PITL/audit_and_convert/logs_after_conversion/NSCLC/conversion_log.csv" \
  --cls-cutoff-years 2.0

# TCGA BLCA (labels from CDR, restricted to imaging patients)
python /home/azureuser/PITL/pre_processing/codes/build_os_labels_multi.py \
  --dataset tcga_blca \
  --cdr-xlsx "/home/azureuser/PITL/clinical_data/TCGA-CDR-SupplementalTableS1.xlsx" \
  --conversion-log "/home/azureuser/PITL/audit_and_convert/logs_after_conversion/TCGA/BLCA/conversion_log.csv" \
  --out-csv "/home/azureuser/PITL/pre_processing/data/TCGA/BLCA/TCGA_BLCA_labels.csv" \
  --cls-cutoff-years 2.0

# Melanoma (OS from staging PET, restricted to imaging patients)
python /home/azureuser/PITL/pre_processing/codes/build_os_labels_multi.py \
  --dataset melanoma \
  --melanoma-xlsx "/home/azureuser/PITL/clinical_data/Cases_PETCT_staging_Melanoma 2.xlsx" \
  --out-csv "/home/azureuser/PITL/pre_processing/data/Melanoma/Melanoma_labels.csv" \
  --conversion-log "/home/azureuser/PITL/audit_and_convert/logs_after_conversion/Melanoma/conversion_log.csv" \
  --cls-cutoff-years 2.0

Supported datasets:
  nsclc,
  tcga_luad, tcga_lusc, tcga_blca, tcga_lihc, tcga_ucec,
  melanoma
"""

import argparse
import os

import numpy as np
import pandas as pd


# ---------- shared KM utilities ----------


def km_median_time(durations, events):
    """
    Kaplan Meier median event time.

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

    uniq_t, idx = np.unique(t, return_index=True)
    N = t.size
    n_at_risk = np.empty_like(uniq_t, dtype=float)
    d_at = np.empty_like(uniq_t, dtype=float)

    for k, s in enumerate(idx):
        e = idx[k + 1] if k + 1 < idx.size else N
        d_at[k] = d[s:e].sum()
        n_at_risk[k] = N - s

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
    Reverse KM to estimate median follow up time.
    Treat censored as events, events as censored.
    """
    return km_median_time(durations, ~np.asarray(events, dtype=bool))


# ---------- shared helpers ----------


def _to_date(series):
    return pd.to_datetime(series, errors="coerce")


def _norm_status(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace())


DEAD_TOKENS = {"dead", "deceased", "expired", "1", "true", "yes", "death", "died"}
ALIVE_TOKENS = {"alive", "living", "0", "false", "no", "censored"}


# ---------- NSCLC branch (copy of original labeling.py logic) ----------


def build_os_labels_nsclc(
    df: pd.DataFrame,
    id_col: str,
    ct_col: str,
    pet_col: str,
    dod_col: str,
    dla_col: str,
    status_col: str,
):
    df = df.copy()

    ct = _to_date(df.get(ct_col))
    pet = _to_date(df.get(pet_col))
    dod = _to_date(df.get(dod_col))
    dla = _to_date(df.get(dla_col))

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

    status_norm = (
        df.get(status_col).map(_norm_status)
        if status_col in df.columns
        else pd.Series("", index=df.index)
    )
    status_dead = status_norm.isin(DEAD_TOKENS)
    event_indicator = ((~dod.isna()) | status_dead).astype(int)

    end_date = dod.fillna(dla)
    event_time_days = (end_date - baseline).dt.days

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
            "patient_id": (
                df[id_col].astype(str)
                if id_col in df.columns
                else np.arange(len(df)).astype(str)
            ),
            "baseline_date": baseline,
            "baseline_source": baseline_source,
            "event_time_days": event_time_days,
            "event_indicator": event_indicator,
            "notes": notes.replace("", np.nan),
        }
    )

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


def run_nsclc(args):
    if args.nsclc_csv is None:
        raise SystemExit("--nsclc-csv is required for dataset=nsclc")

    df = pd.read_csv(args.nsclc_csv)

    # Optional filter to imaging patients via conversion_log
    if args.conversion_log is not None:
        conv = pd.read_csv(args.conversion_log)
        if "patient_id" not in conv.columns:
            raise SystemExit(
                f"'patient_id' column not found in conversion_log {args.conversion_log}. "
                f"Columns: {list(conv.columns)}"
            )
        keep = set(conv["patient_id"].astype(str))
        if args.id_col not in df.columns:
            raise SystemExit(
                f"ID column '{args.id_col}' not found in NSCLC CSV. "
                f"Available: {list(df.columns)[:10]}..."
            )
        df[args.id_col] = df[args.id_col].astype(str)
        before = len(df)
        df = df[df[args.id_col].isin(keep)].copy()
        after = len(df)
        print(f"[info] NSCLC conversion_log filter: {before} -> {after} rows")

    labels, summary = build_os_labels_nsclc(
        df,
        id_col=args.id_col,
        ct_col=args.ct_col,
        pet_col=args.pet_col,
        dod_col=args.dod_col,
        dla_col=args.dla_col,
        status_col=args.status_col,
    )

    cutoff_days = int(round(args.cls_cutoff_years * 365.25))
    cls = (
        (labels["event_indicator"] == 1) & (labels["event_time_days"] <= cutoff_days)
    ).astype(int)
    labels["cls_label"] = cls
    labels["cls_cutoff_days"] = cutoff_days

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    labels.to_csv(args.out_csv, index=False)

    print("=== NSCLC OS Label Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"cls_cutoff_years: {args.cls_cutoff_years}")
    print(f"cls_cutoff_days:  {cutoff_days}")
    print("[info] wrote NSCLC labels to:", args.out_csv)


# ---------- TCGA branch (CDR based) ----------


def load_cdr_table(path):
    """
    Read TCGA-CDR-SupplementalTableS1.xlsx and return a DataFrame.

    If there are multiple sheets, prefer the one whose name contains 'CDR' or 'TCGA'.
    """
    try:
        return pd.read_excel(path)
    except ValueError:
        xls = pd.ExcelFile(path)
        sheet_names = xls.sheet_names
        sheet = sheet_names[0]
        for name in sheet_names:
            up = name.upper()
            if "CDR" in up or "TCGA" in up:
                sheet = name
                break
        return pd.read_excel(path, sheet_name=sheet)


def run_tcga(args, project_short: str):
    if args.cdr_xlsx is None:
        raise SystemExit("--cdr-xlsx is required for TCGA datasets")

    cdr = load_cdr_table(args.cdr_xlsx)
    cols = {c.lower(): c for c in cdr.columns}

    # Required columns in the CDR sheet
    for needed in ["bcr_patient_barcode", "type", "os.time", "vital_status"]:
        if needed not in cols:
            raise SystemExit(
                f"Column '{needed}' not found in CDR sheet. "
                f"Available: {list(cdr.columns)[:10]}..."
            )

    barcode_col = cols["bcr_patient_barcode"]
    type_col = cols["type"]
    ostime_col = cols["os.time"]
    vital_col = cols["vital_status"]

    proj = project_short.upper()
    t = cdr[type_col].astype(str).str.upper()
    mask = (t == proj) | (t == f"TCGA-{proj}")
    cdr_proj = cdr[mask].copy()
    print(
        f"[info] CDR project filter '{proj}': {cdr.shape[0]} -> {cdr_proj.shape[0]} rows"
    )

    patient_id = cdr_proj[barcode_col].astype(str)

    os_time = pd.to_numeric(cdr_proj[ostime_col], errors="coerce")

    vs_raw = cdr_proj[vital_col].astype(str).str.strip().str.lower()
    event_indicator = vs_raw.eq("dead").astype(int)

    labels = pd.DataFrame(
        {
            "patient_id": patient_id,
            "event_time_days": os_time,
            "event_indicator": event_indicator,
        }
    )

    # Drop missing or non positive times
    before = len(labels)
    labels = labels[
        np.isfinite(labels["event_time_days"]) & (labels["event_time_days"] > 0)
    ].copy()
    after = len(labels)
    if after < before:
        print(
            f"[info] dropped {before - after} rows with missing/invalid OS.time "
            f"for project {proj}"
        )

    # Optional restriction to imaging patients
    if args.conversion_log is not None:
        conv = pd.read_csv(args.conversion_log)
        conv_id_col = None
        if "patient_id" in conv.columns:
            conv_id_col = "patient_id"
        elif "case_id" in conv.columns:
            conv_id_col = "case_id"
        else:
            raise SystemExit(
                f"No 'patient_id' or 'case_id' column in conversion_log "
                f"{args.conversion_log}. Columns: {list(conv.columns)}"
            )

        conv_ids = set(conv[conv_id_col].astype(str))
        labels_before = len(labels)
        labels = labels[labels["patient_id"].astype(str).isin(conv_ids)].copy()
        labels_after = len(labels)
        print(
            f"[info] TCGA {proj} conversion_log filter: "
            f"{labels_before} -> {labels_after} rows"
        )

    # KM summary
    N = len(labels)
    events = int(labels["event_indicator"].sum())
    censored = N - events
    censor_frac = censored / N if N > 0 else np.nan
    med_os, os_reached = km_median_time(
        labels["event_time_days"].values.astype(float),
        labels["event_indicator"].values.astype(bool),
    )
    med_fu, fu_reached = reverse_km_median_followup(
        labels["event_time_days"].values.astype(float),
        labels["event_indicator"].values.astype(bool),
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
    }

    cutoff_days = int(round(args.cls_cutoff_years * 365.25))
    cls = (
        (labels["event_indicator"] == 1) & (labels["event_time_days"] <= cutoff_days)
    ).astype(int)
    labels["cls_label"] = cls
    labels["cls_cutoff_days"] = cutoff_days

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    labels.to_csv(args.out_csv, index=False)

    print(f"=== TCGA {proj} OS Label Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"cls_cutoff_years: {args.cls_cutoff_years}")
    print(f"cls_cutoff_days:  {cutoff_days}")
    print("[info] wrote TCGA labels to:", args.out_csv)


# ---------- Melanoma branch (staging PET baseline) ----------


def build_os_labels_melanoma(
    df: pd.DataFrame,
    id_col: str,
    pet_date_col: str,
    end_date_col: str,
    event_col: str,
):
    """
    Build OS labels for melanoma with baseline = staging PET/CT date.

    baseline_date      = date_staging_pet
    event_time_days    = date_of_death_last_fu - date_staging_pet
    event_indicator    = 1 if event == 'dead', 0 otherwise
    baseline_source    = 'STAGING_PET' where baseline is present
    """
    df = df.copy()

    if pet_date_col not in df.columns:
        raise SystemExit(
            f"Melanoma PET baseline column '{pet_date_col}' not in table. "
            f"Columns: {list(df.columns)[:10]}..."
        )
    if end_date_col not in df.columns:
        raise SystemExit(
            f"Melanoma end date column '{end_date_col}' not in table. "
            f"Columns: {list(df.columns)[:10]}..."
        )
    if id_col not in df.columns:
        raise SystemExit(
            f"Melanoma ID column '{id_col}' not in table. "
            f"Columns: {list(df.columns)[:10]}..."
        )

    baseline = _to_date(df[pet_date_col])
    end_date = _to_date(df[end_date_col])

    # event: 'alive' or 'dead'
    if event_col in df.columns:
        ev_raw = df[event_col].astype(str).str.strip().str.lower()
    else:
        ev_raw = pd.Series("", index=df.index)

    event_indicator = ev_raw.eq("dead").astype(int)

    event_time_days = (end_date - baseline).dt.days

    baseline_source = pd.Series("MISSING", index=df.index, dtype=object)
    baseline_source[baseline.notna()] = "STAGING_PET"

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
            "patient_id": df[id_col].astype(str),
            "baseline_date": baseline,
            "baseline_source": baseline_source,
            "event_time_days": event_time_days,
            "event_indicator": event_indicator,
            "notes": notes.replace("", np.nan),
        }
    )

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


def run_melanoma(args):
    if args.melanoma_xlsx is None:
        raise SystemExit("--melanoma-xlsx is required for dataset=melanoma")

    # Read the Excel melanoma clinical table
    df = pd.read_excel(args.melanoma_xlsx)

    # Optional restriction to imaging patients using conversion_log
    # conversion_log for Melanoma has a 'patient' column with IDs like 'mela_001'
    # clinical table has an 'ImageIDs' column with the same IDs
    if args.conversion_log is not None:
        conv = pd.read_csv(args.conversion_log)
        if "patient" not in conv.columns:
            raise SystemExit(
                f"'patient' column not found in Melanoma conversion_log "
                f"{args.conversion_log}. Columns: {list(conv.columns)}"
            )
        keep = set(conv["patient"].astype(str))
        if args.mel_id_col not in df.columns:
            raise SystemExit(
                f"Melanoma ID column '{args.mel_id_col}' not found in clinical table. "
                f"Columns: {list(df.columns)[:10]}..."
            )
        df[args.mel_id_col] = df[args.mel_id_col].astype(str)
        before = len(df)
        df = df[df[args.mel_id_col].isin(keep)].copy()
        after = len(df)
        print(f"[info] Melanoma conversion_log filter: {before} -> {after} rows")

    labels, summary = build_os_labels_melanoma(
        df,
        id_col=args.mel_id_col,
        pet_date_col=args.mel_pet_date_col,
        end_date_col=args.mel_end_date_col,
        event_col=args.mel_event_col,
    )

    cutoff_days = int(round(args.cls_cutoff_years * 365.25))
    cls = (
        (labels["event_indicator"] == 1) & (labels["event_time_days"] <= cutoff_days)
    ).astype(int)
    labels["cls_label"] = cls
    labels["cls_cutoff_days"] = cutoff_days

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    labels.to_csv(args.out_csv, index=False)

    print("=== Melanoma OS Label Summary (baseline = staging PET) ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"cls_cutoff_years: {args.cls_cutoff_years}")
    print(f"cls_cutoff_days:  {cutoff_days}")
    print("[info] wrote Melanoma labels to:", args.out_csv)


# ---------- main ----------


def main():
    p = argparse.ArgumentParser(
        description="Unified OS label builder for NSCLC, TCGA, and Melanoma."
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=[
            "nsclc",
            "tcga_luad",
            "tcga_lusc",
            "tcga_blca",
            "tcga_lihc",
            "tcga_ucec",
            "melanoma",
        ],
    )

    # NSCLC specific
    p.add_argument("--nsclc-csv", help="NSCLC clinical CSV (original 2018 file)")
    p.add_argument("--id-col", default="Case ID")
    p.add_argument("--ct-col", default="CT Date")
    p.add_argument("--pet-col", default="PET Date")
    p.add_argument("--dod-col", default="Date of Death")
    p.add_argument("--dla-col", default="Date of Last Known Alive")
    p.add_argument("--status-col", default="Survival Status")

    # TCGA specific
    p.add_argument("--cdr-xlsx", help="Path to TCGA-CDR-SupplementalTableS1.xlsx")

    # Melanoma specific
    p.add_argument(
        "--melanoma-xlsx",
        help="Path to Cases_PETCT_staging_Melanoma*.xlsx clinical table",
    )
    p.add_argument(
        "--mel-id-col",
        default="ImageIDs",
        help="Melanoma ID column that matches imaging IDs (for example 'ImageIDs')",
    )
    p.add_argument(
        "--mel-pet-date-col",
        default="date_staging_pet",
        help="Melanoma baseline PET date column",
    )
    p.add_argument(
        "--mel-end-date-col",
        dest="mel_end_date_col",
        default="date_of_death_last_fu",
        help="Melanoma end date column (death or last follow up)",
    )
    p.add_argument(
        "--mel-event-col",
        default="event",
        help="Melanoma status column (values like 'alive' or 'dead')",
    )

    # Shared
    p.add_argument(
        "--conversion-log",
        default=None,
        help=(
            "Optional conversion_log.csv path to restrict to imaging patients. "
            "For NSCLC and TCGA expect 'patient_id' or 'case_id'. "
            "For Melanoma expect 'patient' (mela_XXX)."
        ),
    )
    p.add_argument("--out-csv", required=True, help="Output labels CSV")
    p.add_argument(
        "--cls-cutoff-years",
        type=float,
        default=3.0,
        help="Horizon for cls_label (years). Default: 3.0",
    )

    args = p.parse_args()

    if args.dataset == "nsclc":
        run_nsclc(args)
    elif args.dataset == "melanoma":
        run_melanoma(args)
    else:
        proj_map = {
            "tcga_luad": "LUAD",
            "tcga_lusc": "LUSC",
            "tcga_blca": "BLCA",
            "tcga_lihc": "LIHC",
            "tcga_ucec": "UCEC",
        }
        proj = proj_map[args.dataset]
        run_tcga(args, project_short=proj)


if __name__ == "__main__":
    main()
