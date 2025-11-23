"""
python /home/azureuser/PITL/pre_processing/codes/build_melanoma_paths.py
"""

import os
import re
import pandas as pd

# Adjust these if needed
LABELS_CSV = "/home/azureuser/PITL/pre_processing/data/Melanoma/Melanoma_labels.csv"
NIFTI_DIR = "/data/Melanoma_nifti"
OUT_CSV = "/home/azureuser/PITL/pre_processing/data/Melanoma/Melanoma_paths.csv"


def main():
    # 1) Load labels to know which patients we care about
    labels = pd.read_csv(LABELS_CSV)
    if "patient_id" not in labels.columns:
        raise SystemExit(f"'patient_id' column not found in {LABELS_CSV}")
    patients = labels["patient_id"].astype(str).unique()

    # Initialize rows for all labeled patients
    rows = {
        pid: {"patient_id": pid, "ct_path": None, "pet_path": None} for pid in patients
    }

    # 2) Scan NIfTI directory and assign CT / PET based on filename
    # Expected pattern: mela_XXX_CT.nii.gz or mela_XXX_PT.nii.gz
    pattern = re.compile(r"^(mela_\d+)_([A-Za-z]+)\.nii\.gz$")

    for fname in os.listdir(NIFTI_DIR):
        if not fname.endswith(".nii.gz"):
            continue

        m = pattern.match(fname)
        if not m:
            # Not in expected pattern, ignore but print warning
            print(f"[warn] skipping unexpected file name: {fname}")
            continue

        pid, mod = m.groups()
        if pid not in rows:
            # Image exists but patient not in labels -> ignore
            continue

        full_path = os.path.join(NIFTI_DIR, fname)
        mod_up = mod.upper()

        if mod_up == "CT":
            if rows[pid]["ct_path"] is not None:
                print(f"[warn] multiple CT for {pid}, keeping first")
            else:
                rows[pid]["ct_path"] = full_path

        elif mod_up in ("PT", "PET"):
            # If you *do* see mela_001_PT.nii.gz and know it's NAC, delete or rename
            if rows[pid]["pet_path"] is not None:
                print(f"[warn] multiple PET for {pid}, keeping first")
            else:
                rows[pid]["pet_path"] = full_path
        else:
            print(f"[warn] unknown modality '{mod}' in file {fname}")

    paths_df = pd.DataFrame(list(rows.values()))
    paths_df.to_csv(OUT_CSV, index=False)
    print(f"[info] Wrote Melanoma paths to {OUT_CSV}")
    print(paths_df.head())


if __name__ == "__main__":
    main()
