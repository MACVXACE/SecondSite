import os
import numpy as np
import pandas as pd

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
HOSPITAL_FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "Long Beach VA": "processed.va.data",
}

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]

# 7 routine checkup measurements available at a district outpatient clinic
CHECKUP_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"]

# 6 stress-test / invasive imaging measurements requiring specialist equipment
STRESS_IMAGING_FEATURES = ["thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def load_multi_hospital_data(cache_dir: str = "data") -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    frames = []

    for hospital, filename in HOSPITAL_FILES.items():
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            df_h = pd.read_csv(
                UCI_BASE_URL + filename,
                header=None,
                names=COLUMNS,
                na_values=["?", -9.0],
            )
            df_h.to_csv(local_path, index=False)
        else:
            df_h = pd.read_csv(local_path)

        df_h["hospital"] = hospital
        frames.append(df_h)

    df = pd.concat(frames, ignore_index=True)

    # Fix physiological zero-encoding artifacts (e.g., chol=0 in Switzerland/VA means missing)
    df.loc[df["trestbps"] <= 0, "trestbps"] = np.nan
    df.loc[df["chol"] <= 0, "chol"] = np.nan

    # Binary target per Brief 4.1.2: 0 = no disease, 1 = any coronary narrowing (>50%)
    df["target"] = (df["num"] > 0).astype(int)
    return df


def run_leakage_and_missingness_audit(df: pd.DataFrame) -> pd.DataFrame:
    audit_rows = []
    for hospital, group in df.groupby("hospital", sort=False):
        row = {
            "Hospital": hospital,
            "Patients (N)": len(group),
            "Disease Prevalence": f"{group['target'].mean():.1%}",
            "Missing trestbps": f"{group['trestbps'].isna().mean():.1%}",
            "Missing chol": f"{group['chol'].isna().mean():.1%}",
            "Missing fbs": f"{group['fbs'].isna().mean():.1%}",
            "Missing slope": f"{group['slope'].isna().mean():.1%}",
            "Missing ca": f"{group['ca'].isna().mean():.1%}",
            "Missing thal": f"{group['thal'].isna().mean():.1%}",
        }
        audit_rows.append(row)

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv("artifacts/hospital_missingness_audit.csv", index=False)
    return audit_df


if __name__ == "__main__":
    df = load_multi_hospital_data()
    print(f"Loaded {len(df)} total patient records across {df['hospital'].nunique()} hospitals.\n")
    audit_df = run_leakage_and_missingness_audit(df)
    print("=== HOSPITAL PREVALENCE & MISSINGNESS LEAKAGE AUDIT ===")
    print(audit_df.to_string(index=False))
    print("\nSaved audit table to artifacts/hospital_missingness_audit.csv")
