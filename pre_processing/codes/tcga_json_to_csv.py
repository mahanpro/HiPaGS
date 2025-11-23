"""
python /home/azureuser/PITL/pre_processing/codes/tcga_json_to_csv.py \
  --in-json "/home/azureuser/PITL/clinical_data/clinical.project-tcga-ucec.2025-11-18.json" \
  --out-csv "/home/azureuser/PITL/clinical_data/TCGA_UCEC_clinical.csv"
"""

import argparse, json
import pandas as pd
from pathlib import Path


def pick_primary_diagnosis(diags):
    if not diags:
        return None
    for d in diags:
        if d.get("diagnosis_is_primary_disease") is True:
            return d
    return diags[0]


def extract_row(rec):
    out = {}

    # IDs
    out["case_id"] = rec.get("case_id")
    out["submitter_id"] = rec.get("submitter_id")

    # Demographic
    demo = rec.get("demographic") or {}
    out["age_at_index"] = demo.get("age_at_index")
    # Prefer sex_at_birth if present, fall back to gender
    out["gender"] = demo.get("sex_at_birth") or demo.get("gender")
    out["race"] = demo.get("race")
    out["ethnicity"] = demo.get("ethnicity")
    out["vital_status"] = demo.get("vital_status")
    out["days_to_death"] = demo.get("days_to_death")

    # Diagnosis (primary if possible)
    diags = rec.get("diagnoses") or []
    d0 = pick_primary_diagnosis(diags)
    if d0 is not None:
        out["age_at_diagnosis"] = d0.get("age_at_diagnosis")
        out["ajcc_pathologic_stage"] = d0.get("ajcc_pathologic_stage")
        out["ajcc_pathologic_t"] = d0.get("ajcc_pathologic_t")
        out["ajcc_pathologic_n"] = d0.get("ajcc_pathologic_n")
        out["ajcc_pathologic_m"] = d0.get("ajcc_pathologic_m")
        out["prior_malignancy"] = d0.get("prior_malignancy")
        out["days_to_last_follow_up"] = d0.get("days_to_last_follow_up")

        # Vascular invasion if present (BLCA, LIHC)
        for pd in d0.get("pathology_details") or []:
            if "vascular_invasion_present" in pd:
                out["vascular_invasion_present"] = pd.get("vascular_invasion_present")
                break

    # Tobacco exposure (LUAD, LUSC, BLCA, LIHC)
    exps = rec.get("exposures") or []
    tobacco_exps = [e for e in exps if e.get("exposure_type") == "Tobacco"]
    if tobacco_exps:
        e0 = tobacco_exps[0]
        out["tobacco_smoking_status"] = e0.get("tobacco_smoking_status")
        out["pack_years_smoked"] = e0.get("pack_years_smoked")

    # Follow-ups: BMI and lung function, molecular tests
    followups = rec.get("follow_ups") or []

    # BMI (BLCA, LIHC) and other_clinical_attributes
    bmi_val = None
    dlco_val = None
    for fu in followups:
        for oca in fu.get("other_clinical_attributes") or []:
            if bmi_val is None and oca.get("bmi") is not None:
                bmi_val = oca.get("bmi")
            if dlco_val is None and oca.get("dlco_ref_predictive_percent") is not None:
                dlco_val = oca.get("dlco_ref_predictive_percent")
        if bmi_val is not None and dlco_val is not None:
            break
    if bmi_val is not None:
        out["bmi"] = bmi_val
    if dlco_val is not None:
        out["dlco_ref_predictive_percent"] = dlco_val

    # Molecular tests for lung cohorts: EGFR, KRAS
    egfr_status = None
    kras_status = None
    for fu in followups:
        for mt in fu.get("molecular_tests") or []:
            gene = mt.get("gene_symbol")
            res = mt.get("test_result")
            if gene == "EGFR" and egfr_status is None:
                egfr_status = res
            elif gene == "KRAS" and kras_status is None:
                kras_status = res
        if egfr_status is not None and kras_status is not None:
            break
    if egfr_status is not None:
        out["EGFR_status"] = egfr_status
    if kras_status is not None:
        out["KRAS_status"] = kras_status

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-json",
        required=True,
        help="Path to TCGA clinical JSON file (list of cases)",
    )
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    data = json.load(open(args.in_json, "r"))
    # Some downloads wrap in {"data": [...]} rather than a plain list
    if isinstance(data, dict) and "data" in data:
        records = data["data"]
    elif isinstance(data, list):
        records = data
    else:
        raise SystemExit(f"Unexpected JSON root type: {type(data)}")

    rows = [extract_row(rec) for rec in records]
    df = pd.DataFrame(rows)

    # Choose your patient_id convention: use submitter_id for TCGA
    if "submitter_id" in df.columns:
        df.insert(0, "patient_id", df["submitter_id"])
    else:
        df.insert(0, "patient_id", df["case_id"])

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
