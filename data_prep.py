import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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

CHECKUP_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg"]
STRESS_IMAGING_FEATURES = ["thalach", "exang", "oldpeak", "slope", "ca", "thal"]


class ClinicalCheckupEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X[CHECKUP_FEATURES].copy()
        df["is_hypertensive"] = (df["trestbps"] >= 140).astype(float)
        df["asymptomatic_cp"] = (df["cp"] == 4).astype(float)
        df["age_x_sbp"] = (df["age"] * df["trestbps"].fillna(130.0)) / 100.0
        return df


def load_multi_hospital_data(cache_dir: str = "data") -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    frames = []
    for hospital, filename in HOSPITAL_FILES.items():
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            df_h = pd.read_csv(UCI_BASE_URL + filename, header=None, names=COLUMNS, na_values=["?", -9.0])
            df_h.to_csv(local_path, index=False)
        else:
            df_h = pd.read_csv(local_path)
        df_h["hospital"] = hospital
        frames.append(df_h)

    df = pd.concat(frames, ignore_index=True)
    df.loc[df["trestbps"] <= 0, "trestbps"] = np.nan
    df.loc[df["chol"] <= 0, "chol"] = np.nan
    df["target"] = (df["num"] > 0).astype(int)
    return df
