"""
Build concept labels + masks for NSCLC.

Usage:

python nsclc_concepts_from_ehr.py \
  --ehr-csv "/home/azureuser/PITL/pre_processing/NSCLC_EHR_features.csv" \
  --out-csv "/home/azureuser/PITL/pre_processing/NSCLC_concepts.csv"
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ehr-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.ehr_csv)

    # Expect a 'patient_id' column from ehr_features_NSCLC.py
    if "patient_id" not in df.columns:
        raise SystemExit(
            f"'patient_id' column not found in {args.ehr_csv}. "
            f"Available: {list(df.columns)[:10]}..."
        )

    # Mapping from concept name -> (bin_col, unk_col)
    concept_specs = {
        "EGFR": (
            "EGFR_mutation_status_bin",
            "EGFR_mutation_status_unknown",
        ),
        "KRAS": (
            "KRAS_mutation_status_bin",
            "KRAS_mutation_status_unknown",
        ),
        "ALK": (
            "ALK_translocation_status_bin",
            "ALK_translocation_status_unknown",
        ),
        "Adjuvant": (
            "Adjuvant_Treatment_bin",
            "Adjuvant_Treatment_unknown",
        ),
        "Chemo": (
            "Chemotherapy_bin",
            "Chemotherapy_unknown",
        ),
        "Radiation": (
            "Radiation_bin",
            "Radiation_unknown",
        ),
    }

    out = pd.DataFrame()
    out["patient_id"] = df["patient_id"].astype(str)

    # 6 EHR binary concepts
    for name, (bin_col, unk_col) in concept_specs.items():
        if bin_col not in df.columns or unk_col not in df.columns:
            raise SystemExit(
                f"Expected columns '{bin_col}' and '{unk_col}' in {args.ehr_csv} "
                f"for concept {name}."
            )
        out[f"{name}_label"] = df[bin_col].astype(int)
        out[f"{name}_missing"] = df[unk_col].astype(int)  # 1 = unknown / missing

    # 2 image-derived concepts: TumVolBin, SUVPeakBin
    # For now: mark as missing for everyone. Fill later once you compute them.
    out["TumVolBin_label"] = -1  # placeholder
    out["TumVolBin_missing"] = 1

    out["SUVPeakBin_label"] = -1  # placeholder
    out["SUVPeakBin_missing"] = 1

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote concept labels to {args.out_csv}")
    print("Columns:", list(out.columns))


if __name__ == "__main__":
    main()
