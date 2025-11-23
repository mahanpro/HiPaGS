"""
Unified EHR feature + concept builder for multiple datasets.

Supports:
  - NSCLC Radiogenomics (nsclc)
  - Private Melanoma (melanoma)
  - TCGA: LUAD, LUSC, BLCA, LIHC, UCEC (luad, lusc, blca, lihc, ucec)

Usage example for NSCLC:

python /home/azureuser/PITL/pre_processing/codes/build_ehr_and_concepts.py \
  --dataset nsclc \
  --csv "/data/tcia_out/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv" \
  --out-ehr-csv "/home/azureuser/PITL/pre_processing/data/NSCLC/NSCLC_EHR_features.csv" \
  --out-concept-csv "/home/azureuser/PITL/pre_processing/data/NSCLC/NSCLC_concepts.csv" \
  --id-col "Case ID" \
  --schema-json "/home/azureuser/PITL/pre_processing/data/NSCLC/NSCLC_schema.json"

Usage example for TCGA LUAD:

python /home/azureuser/PITL/pre_processing/codes/build_ehr_and_concepts.py \
  --dataset lusc \
  --csv "/home/azureuser/PITL/clinical_data/TCGA_LUSC_clinical.csv" \
  --out-ehr-csv "/home/azureuser/PITL/pre_processing/data/TCGA/LUSC/TCGA_LUSC_EHR_features.csv" \
  --out-concept-csv "/home/azureuser/PITL/pre_processing/data/TCGA/LUSC/TCGA_LUSC_concepts.csv" \
  --id-col "patient_id" \
  --schema-json "/home/azureuser/PITL/pre_processing/data/TCGA/LUSC/TCGA_LUSC_schema.json"

Melanoma:

python /home/azureuser/PITL/pre_processing/codes/build_ehr_and_concepts.py \
  --dataset melanoma \
  --csv "/home/azureuser/PITL/clinical_data/Cases_PETCT_staging_Melanoma_flat.csv" \
  --out-ehr-csv "/home/azureuser/PITL/pre_processing/data/Melanoma/Melanoma_EHR_features.csv" \
  --out-concept-csv "/home/azureuser/PITL/pre_processing/data/Melanoma/Melanoma_concepts.csv" \
  --id-col "ImageIDs" \
  --pairing-log "/home/azureuser/PITL/pre_processing/data/Melanoma/Melanoma_manifest.csv" \
  --schema-json "/home/azureuser/PITL/pre_processing/data/Melanoma/Melanoma_schema.json"

You must pass a flattened clinical CSV per dataset; project specific
flattening (from GDC JSON to one row per patient) is done elsewhere.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Global concept slots (fixed across datasets)
GLOBAL_CONCEPTS = [
    "AgeBin",
    "Sex",
    "StageGroup",
    "ECOG",
    "SmokingEver",
    "EGFR",
    "KRAS",
    "ALK",
    "BRAF",
    "Adjuvant",
    "Chemo",
    "Radiation",
    "VascularInvasion",
    "TumVolBin",
    "SUVPeakBin",
    "ViralStatus",
]


# ---------- generic helpers ----------

UNK_TOKENS = {
    "",
    "na",
    "n/a",
    "not available",
    "unknown",
    "unk",
    "missing",
    "none",
    "null",
    ".",
}


def _norm_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    # keep alnum and space
    return "".join(ch for ch in s if ch.isalnum() or ch.isspace())


def parse_binary(series, true_tokens, false_tokens):
    """
    Convert a raw categorical column into tri-state binary.

    Returns:
      bin_val: 0 or 1, arbitrary when unknown
      unk: 1 if unknown / missing, 0 if known
    """
    s = series.map(_norm_str)
    is_true = s.isin(true_tokens)
    is_false = s.isin(false_tokens)

    unk = (~is_true & ~is_false).astype(int)
    # default known false, then set true
    bin_val = np.zeros(len(series), dtype=int)
    bin_val[is_true.values] = 1

    return bin_val, unk.values


def zscore(series):
    x = pd.to_numeric(series, errors="coerce").astype(float)
    mask = np.isfinite(x)
    if mask.sum() == 0:
        # nothing valid; return zeros and missing=1
        mu = 0.0
        sd = 1.0
        z = np.zeros_like(x, dtype=float)
        miss = np.ones_like(x, dtype=int)
        return z, miss, mu, sd

    mu = float(x[mask].mean())
    sd = float(x[mask].std(ddof=0))
    if sd <= 0 or not np.isfinite(sd):
        sd = 1.0
    z = (x - mu) / sd
    miss = (~mask).astype(int)
    z[~mask] = 0.0
    return z, miss, mu, sd


def safe_name(col):
    return re.sub(r"[^A-Za-z0-9_]+", "_", col.strip())


# ---------- dataset registry ----------

# NSCLC and melanoma as before

DATASET_CONFIG = {
    "nsclc": {
        "id_col": "Case ID",
        "binary_cols": {
            "EGFR mutation status": (
                {"egfrpositive", "positive"},
                {"egfrnegative", "negative"},
            ),
            "KRAS mutation status": (
                {"kraspositive", "positive"},
                {"krasnegative", "negative"},
            ),
            "ALK translocation status": (
                {"alkpositive", "positive"},
                {"alknegative", "negative"},
            ),
            "Adjuvant Treatment": ({"yes", "y"}, {"no", "n"}),
            "Chemotherapy": ({"yes", "y"}, {"no", "n"}),
            "Radiation": ({"yes", "y"}, {"no", "n"}),
            "Lymphovascular invasion": ({"present", "yes"}, {"absent", "no"}),
        },
        "continuous_cols": [
            "Age at Histological Diagnosis",
            "Weight (lbs)",
            "Pack Years",
            "Days between CT and surgery",
        ],
        "leak_cols": {
            "Survival Status",
            "Date of Death",
            "Time to Death (days)",
            "Date of Last Known Alive",
            "Recurrence",
            "Recurrence Location",
            "Date of Recurrence",
        },
        "concept_map": {
            "AgeBin": ("derive",),
            "Sex": ("derive",),
            "StageGroup": ("derive",),
            "SmokingEver": ("derive",),
            "EGFR": ("bin", "EGFR_mutation_status"),
            "KRAS": ("bin", "KRAS_mutation_status"),
            "ALK": ("bin", "ALK_translocation_status"),
            "Adjuvant": ("bin", "Adjuvant_Treatment"),
            "Chemo": ("bin", "Chemotherapy"),
            "Radiation": ("bin", "Radiation"),
            "VascularInvasion": ("bin", "Lymphovascular_invasion"),
            "TumVolBin": ("placeholder",),
            "SUVPeakBin": ("placeholder",),
            # BRAF, ECOG, ViralStatus absent
        },
    },
    "melanoma": {
        "id_col": "bcca_id",
        "binary_cols": {
            "braf_mut": ({"mutant", "positive", "yes"}, {"wildtype", "negative", "no"}),
            "autoimmune_hx": ({"yes", "y"}, {"no", "n"}),
            "brain_mets": ({"yes", "y", "present"}, {"no", "n", "absent"}),
        },
        "continuous_cols": [
            "age_at_dx_years",  # you compute this before calling the script
            "breslow",
            "albumin",
            "ldh",
            "ldh_uln",
            "alc",
            "anc",
        ],
        "leak_cols": {
            "date_of_death_last_fu",
            "event",
            "cause_of_death",
            "date_of_mets",
            "site_of_mets",
        },
        "concept_map": {
            "AgeBin": ("derive",),
            "Sex": ("derive",),
            "StageGroup": ("derive",),
            "ECOG": ("derive",),
            "BRAF": ("bin", "braf_mut"),
            "TumVolBin": ("placeholder",),
            "SUVPeakBin": ("placeholder",),
        },
    },
}

# Shared TCGA config for LUAD, LUSC, BLCA, LIHC, UCEC
TCGA_COMMON_CONFIG = {
    # matches tcga_json_to_csv.py output
    "id_col": "patient_id",
    "binary_cols": {
        # tobacco; used as EHR feature only, SmokingEver is derived at concept level
        "tobacco_smoking_status": (
            {
                "current smoker",
                "currentsmoker",
                "former smoker",
                "formersmoker",
            },
            {
                "never smoker",
                "neversmoker",
                "lifelong nonsmoker",
                "lifelongnonsmoker",
            },
        ),
        "vascular_invasion_present": (
            {"yes", "present"},
            {"no", "absent", "not present"},
        ),
        "EGFR_status": (
            {"positive", "pos", "mutated", "mutation detected"},
            {"negative", "neg", "wildtype", "no mutation detected"},
        ),
        "KRAS_status": (
            {"positive", "pos", "mutated", "mutation detected"},
            {"negative", "neg", "wildtype", "no mutation detected"},
        ),
        "prior_malignancy": ({"yes", "y"}, {"no", "n"}),
    },
    "continuous_cols": [
        "age_at_index",
        "age_at_diagnosis",
        "pack_years_smoked",
        "bmi",
        "dlco_ref_predictive_percent",
    ],
    "leak_cols": {
        "vital_status",
        "days_to_death",
        "days_to_last_follow_up",
    },
    "concept_map": {
        "AgeBin": ("derive",),
        "Sex": ("derive",),
        "StageGroup": ("derive",),
        "SmokingEver": ("derive",),
        "EGFR": ("bin", "EGFR_status"),
        "KRAS": ("bin", "KRAS_status"),
        "VascularInvasion": ("bin", "vascular_invasion_present"),
        # TumVolBin, SUVPeakBin, etc. are imaging derived later
    },
}

# Register TCGA datasets
for _tcga_name in ["luad", "lusc", "blca", "lihc", "ucec"]:
    DATASET_CONFIG[_tcga_name] = TCGA_COMMON_CONFIG


# ---------- core EHR feature builder ----------


def build_ehr_features(df, cfg, id_col_override=None):
    df = df.copy()

    id_col = id_col_override or cfg["id_col"]
    if id_col not in df.columns:
        raise SystemExit(
            f"ID column '{id_col}' not in CSV. Available: {list(df.columns)[:10]}..."
        )

    df[id_col] = df[id_col].astype(str)
    ehr = pd.DataFrame()
    ehr["patient_id"] = df[id_col].astype(str)

    schema = {"binary": {}, "continuous": {}, "dataset_id_col": id_col}

    # Binary columns
    for raw_col, (true_tokens, false_tokens) in cfg.get("binary_cols", {}).items():
        if raw_col not in df.columns:
            print(f"[warn] binary column '{raw_col}' not in CSV, skipping")
            continue
        bin_val, unk = parse_binary(df[raw_col], true_tokens, false_tokens)
        base = safe_name(raw_col)
        bin_name = f"{base}_bin"
        unk_name = f"{base}_unknown"
        ehr[bin_name] = bin_val.astype(int)
        ehr[unk_name] = unk.astype(int)
        schema["binary"][raw_col] = {
            "bin_col": bin_name,
            "unknown_col": unk_name,
            "true_tokens": sorted(list(true_tokens)),
            "false_tokens": sorted(list(false_tokens)),
        }

    # Continuous columns
    schema["continuous"]["cols"] = {}
    for raw_col in cfg.get("continuous_cols", []):
        if raw_col not in df.columns:
            print(f"[warn] continuous column '{raw_col}' not in CSV, skipping")
            continue
        z, miss, mu, sd = zscore(df[raw_col])
        base = safe_name(raw_col)
        z_name = f"{base}_z"
        miss_name = f"{base}_missing"
        ehr[z_name] = z.astype(float)
        ehr[miss_name] = miss.astype(int)
        schema["continuous"]["cols"][raw_col] = {
            "z_col": z_name,
            "missing_col": miss_name,
            "mu": mu,
            "sd": sd,
        }

    return ehr, schema


# ---------- concept builder ----------


def bin_age(age_series):
    """
    Simple age binning: <60 -> 0, 60-70 -> 1, >70 -> 2.
    """
    x = pd.to_numeric(age_series, errors="coerce").astype(float)
    miss = (~np.isfinite(x)).astype(int)
    labels = np.full(len(x), -1, dtype=int)
    labels[(x < 60) & np.isfinite(x)] = 0
    labels[(x >= 60) & (x < 70)] = 1
    labels[(x >= 70)] = 2
    return labels, miss


def build_concepts(dataset, ehr_df, cfg, raw_df):
    """
    Build concept DataFrame from EHR features and raw df.

    Output columns:
      patient_id,
      <Concept>_label,
      <Concept>_missing
    for all GLOBAL_CONCEPTS.
    """
    out = pd.DataFrame()
    out["patient_id"] = ehr_df["patient_id"].astype(str)

    # Initialize all concepts as missing
    for name in GLOBAL_CONCEPTS:
        out[f"{name}_label"] = -1
        out[f"{name}_missing"] = 1

    cmap = cfg.get("concept_map", {})

    # Dataset specific logic
    if dataset == "nsclc":
        # AgeBin from raw age
        if "Age at Histological Diagnosis" in raw_df.columns:
            labels, miss = bin_age(raw_df["Age at Histological Diagnosis"])
            out["AgeBin_label"] = labels
            out["AgeBin_missing"] = miss

        # Sex from Gender (0 = female, 1 = male)
        if "Gender" in raw_df.columns:
            s = raw_df["Gender"].map(_norm_str)
            lab = np.full(len(s), -1, dtype=int)
            miss = np.ones(len(s), dtype=int)
            is_m = s.isin({"male", "m"})
            is_f = s.isin({"female", "f"})
            lab[is_f.values] = 0
            lab[is_m.values] = 1
            miss[(is_m | is_f).values] = 0
            out["Sex_label"] = lab
            out["Sex_missing"] = miss

        # StageGroup from pathologic TNM or explicit stage
        stage_cols = [
            c for c in raw_df.columns if "Pathological" in c and "stage" in c.lower()
        ]
        if stage_cols:
            col = stage_cols[0]
            s = raw_df[col].map(_norm_str)
            lab = np.full(len(s), -1, dtype=int)
            miss = np.ones(len(s), dtype=int)
            # crude mapping: I ->0, II->1, III->2, IV->3
            for idx, val in enumerate(s):
                if not val:
                    continue
                if "iv" in val:
                    lab[idx] = 3
                elif "iii" in val:
                    lab[idx] = 2
                elif "ii" in val:
                    lab[idx] = 1
                elif "i" in val:
                    lab[idx] = 0
                else:
                    continue
                miss[idx] = 0
            out["StageGroup_label"] = lab
            out["StageGroup_missing"] = miss

        # SmokingEver from Smoking status + Pack Years
        if "Smoking status" in raw_df.columns or "Pack Years" in raw_df.columns:
            if "Smoking status" in raw_df.columns:
                s = raw_df["Smoking status"].map(_norm_str)
            else:
                s = pd.Series([""] * len(raw_df))
            py = pd.to_numeric(raw_df.get("Pack Years", np.nan), errors="coerce")
            ever = s.isin({"current", "former", "ever smoker"}) | (py.fillna(0) > 0)
            never = s.isin({"never", "never smoker"})
            lab = np.full(len(s), -1, dtype=int)
            miss = np.ones(len(s), dtype=int)
            lab[never.values] = 0
            lab[ever.values] = 1
            miss[(never | ever).values] = 0
            out["SmokingEver_label"] = lab
            out["SmokingEver_missing"] = miss

    elif dataset == "melanoma":
        # AgeBin from age_at_dx_years
        if "age_at_dx_years" in raw_df.columns:
            labels, miss = bin_age(raw_df["age_at_dx_years"])
            out["AgeBin_label"] = labels
            out["AgeBin_missing"] = miss

        # Sex from gender
        if "gender" in raw_df.columns:
            s = raw_df["gender"].map(_norm_str)
            lab = np.full(len(s), -1, dtype=int)
            miss = np.ones(len(s), dtype=int)
            is_m = s.isin({"male", "m"})
            is_f = s.isin({"female", "f"})
            lab[is_f.values] = 0
            lab[is_m.values] = 1
            miss[(is_m | is_f).values] = 0
            out["Sex_label"] = lab
            out["Sex_missing"] = miss

        # ECOG
        if "ecog" in raw_df.columns:
            ec = pd.to_numeric(raw_df["ecog"], errors="coerce").astype(float)
            lab = np.full(len(ec), -1, dtype=int)
            miss = np.ones(len(ec), dtype=int)
            valid = np.isfinite(ec)
            lab[valid] = ec[valid].astype(int)
            miss[valid] = 0
            out["ECOG_label"] = lab
            out["ECOG_missing"] = miss

        # StageGroup: can be added later from T/N/M or AJCC stage

    elif dataset in {"luad", "lusc", "blca", "lihc", "ucec"}:
        # AgeBin from age_at_index (preferred) or age_at_diagnosis
        if "age_at_index" in raw_df.columns:
            labels, miss = bin_age(raw_df["age_at_index"])
            out["AgeBin_label"] = labels
            out["AgeBin_missing"] = miss
        elif "age_at_diagnosis" in raw_df.columns:
            labels, miss = bin_age(raw_df["age_at_diagnosis"])
            out["AgeBin_label"] = labels
            out["AgeBin_missing"] = miss

        # Sex from gender / sex_at_birth (normalized to 'gender' in tcga_json_to_csv)
        for sex_col in ["gender", "sex"]:
            if sex_col in raw_df.columns:
                s = raw_df[sex_col].map(_norm_str)
                lab = np.full(len(s), -1, dtype=int)
                miss = np.ones(len(s), dtype=int)
                is_m = s.isin({"male", "m"})
                is_f = s.isin({"female", "f"})
                lab[is_f.values] = 0
                lab[is_m.values] = 1
                miss[(is_m | is_f).values] = 0
                out["Sex_label"] = lab
                out["Sex_missing"] = miss
                break

        # StageGroup from AJCC pathologic stage
        stage_cols = [
            c for c in raw_df.columns if "stage" in c.lower() and "ajcc" in c.lower()
        ]
        if stage_cols:
            col = stage_cols[0]
            s = raw_df[col].map(_norm_str)
            lab = np.full(len(s), -1, dtype=int)
            miss = np.ones(len(s), dtype=int)
            for idx, val in enumerate(s):
                if not val:
                    continue
                if "iv" in val:
                    lab[idx] = 3
                elif "iii" in val:
                    lab[idx] = 2
                elif "ii" in val:
                    lab[idx] = 1
                elif "i" in val:
                    lab[idx] = 0
                else:
                    continue
                miss[idx] = 0
            out["StageGroup_label"] = lab
            out["StageGroup_missing"] = miss

        # SmokingEver from tobacco_smoking_status + pack_years_smoked
        if (
            "tobacco_smoking_status" in raw_df.columns
            or "pack_years_smoked" in raw_df.columns
        ):
            if "tobacco_smoking_status" in raw_df.columns:
                s = raw_df["tobacco_smoking_status"].map(_norm_str)
            else:
                s = pd.Series([""] * len(raw_df))
            py = pd.to_numeric(raw_df.get("pack_years_smoked", np.nan), errors="coerce")

            # never: contains "never" or "nonsmoker"
            never = s.str.contains("never") | s.str.contains("nonsmoker")
            # ever: smoker terms excluding the above, or positive pack years
            ever_smoke = (~never) & s.str.contains("smoker")
            ever_pack = py.fillna(0) > 0
            ever = ever_smoke | ever_pack

            lab = np.full(len(s), -1, dtype=int)
            miss = np.ones(len(s), dtype=int)
            lab[never.values] = 0
            lab[ever.values] = 1
            miss[(never | ever).values] = 0
            out["SmokingEver_label"] = lab
            out["SmokingEver_missing"] = miss

    # Fill binary-driven concepts based on EHR _bin/_unknown columns
    for concept, spec in cmap.items():
        if concept not in GLOBAL_CONCEPTS:
            continue
        if spec[0] == "bin" and len(spec) == 2:
            raw_name = spec[1]
            base = safe_name(raw_name)
            bin_col = f"{base}_bin"
            unk_col = f"{base}_unknown"
            if bin_col in ehr_df.columns and unk_col in ehr_df.columns:
                out[f"{concept}_label"] = ehr_df[bin_col].astype(int)
                out[f"{concept}_missing"] = ehr_df[unk_col].astype(int)

        elif spec[0] == "placeholder":
            # Already initialized as missing; nothing to do
            continue

    return out


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    ap.add_argument(
        "--csv", required=True, help="Flattened clinical CSV for this dataset"
    )
    ap.add_argument("--out-ehr-csv", required=True)
    ap.add_argument("--out-concept-csv", required=True)
    ap.add_argument(
        "--id-col", default=None, help="Override ID column name from config"
    )
    ap.add_argument(
        "--pairing-log",
        default=None,
        help="Optional pairing_log.csv to restrict to imaging-paired patients",
    )
    ap.add_argument(
        "--schema-json", default=None, help="Optional path to write EHR schema json"
    )
    args = ap.parse_args()

    cfg = DATASET_CONFIG[args.dataset]
    df = pd.read_csv(args.csv)

    # Optional filtering using pairing_log for imaging datasets
    if args.pairing_log is not None:
        pl = pd.read_csv(args.pairing_log)
        if "patient_id" not in pl.columns:
            raise SystemExit(
                f"'patient_id' column not found in pairing_log {args.pairing_log}, "
                f"columns are {list(pl.columns)}"
            )
        keep = set(pl["patient_id"].astype(str))
        id_col = args.id_col or cfg["id_col"]
        if id_col not in df.columns:
            raise SystemExit(
                f"ID column '{id_col}' not found in clinical CSV. "
                f"Available: {list(df.columns)[:10]}..."
            )
        df[id_col] = df[id_col].astype(str)
        before = len(df)
        df = df[df[id_col].isin(keep)].copy()
        after = len(df)
        print(f"[info] pairing_log filter: {before} -> {after} rows")

    # Drop leak columns if present
    for col in cfg.get("leak_cols", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    ehr_df, schema = build_ehr_features(df, cfg, id_col_override=args.id_col)
    ehr_df.to_csv(args.out_ehr_csv, index=False)
    print(f"[info] wrote EHR features to {args.out_ehr_csv}")

    if args.schema_json:
        Path(args.schema_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.schema_json, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"[info] wrote schema json to {args.schema_json}")

    concepts_df = build_concepts(args.dataset, ehr_df, cfg, df)
    concepts_df.to_csv(args.out_concept_csv, index=False)
    print(f"[info] wrote concepts to {args.out_concept_csv}")
    print(
        "[info] concept columns:",
        [c for c in concepts_df.columns if c.endswith("_label")],
    )


if __name__ == "__main__":
    main()
