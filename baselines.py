import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from data_prep import (
    load_multi_hospital_data,
    CHECKUP_FEATURES,
    STRESS_IMAGING_FEATURES,
)


def build_preprocessor(feature_list):
    cat_candidates = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    cat_cols = [c for c in feature_list if c in cat_candidates]
    num_cols = [c for c in feature_list if c not in cat_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def score_cohort(label: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "Cohort / Model": label,
        "N": len(y_true),
        "Prevalence": round(float(y_true.mean()), 3),
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Sensitivity (Recall)": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "Specificity": round(specificity, 3),
        "Missed Cases (FN)": int(fn),
        "False Alarms (FP)": int(fp),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 3),
        "PR-AUC": round(average_precision_score(y_true, y_prob), 3),
        "Brier Score": round(brier_score_loss(y_true, y_prob), 3),
    }


if __name__ == "__main__":
    df = load_multi_hospital_data()
    cleveland = df[df["hospital"] == "Cleveland"].copy()

    # Brief 4.1.2: 80/20 stratified split on Cleveland
    train_df, test_df = train_test_split(
        cleveland, test_size=0.20, stratify=cleveland["target"], random_state=42
    )

    results = []

    # Tier 0: Majority-class baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df[CHECKUP_FEATURES], train_df["target"])
    results.append(
        score_cohort(
            "Tier 0: Majority Baseline (Cleveland Test)",
            test_df["target"].values,
            dummy.predict_proba(test_df[CHECKUP_FEATURES])[:, 1],
        )
    )

    # Tier 1A: Brief 4.1.2 Standard 13-Feature Logistic Regression on Cleveland
    all_13 = CHECKUP_FEATURES + STRESS_IMAGING_FEATURES
    lr_13 = Pipeline([
        ("prep", build_preprocessor(all_13)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr_13.fit(train_df[all_13], train_df["target"])
    results.append(
        score_cohort(
            "Tier 1A: 13-Feature LR (Cleveland Test)",
            test_df["target"].values,
            lr_13.predict_proba(test_df[all_13])[:, 1],
        )
    )

    # Tier 1B: 7-Feature Checkup-Only Logistic Regression (District Clinic Realistic)
    lr_7 = Pipeline([
        ("prep", build_preprocessor(CHECKUP_FEATURES)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    lr_7.fit(train_df[CHECKUP_FEATURES], train_df["target"])
    results.append(
        score_cohort(
            "Tier 1B: 7-Feature Checkup LR (Cleveland Test)",
            test_df["target"].values,
            lr_7.predict_proba(test_df[CHECKUP_FEATURES])[:, 1],
        )
    )

    # Cross-Hospital Zero-Shot Drop (Unseen Hospitals)
    for hosp in ["Hungary", "Switzerland", "Long Beach VA"]:
        ext = df[df["hospital"] == hosp]
        results.append(
            score_cohort(
                f"Unseen Hospital ({hosp}) — 7-Feat LR",
                ext["target"].values,
                lr_7.predict_proba(ext[CHECKUP_FEATURES])[:, 1],
            )
        )

    res_df = pd.DataFrame(results)
    res_df.to_csv("artifacts/tier0_tier1_cross_hospital.csv", index=False)
    print("=== TIER 0, TIER 1 & CROSS-HOSPITAL GENERALIZATION BENCHMARK ===")
    print(res_df.to_string(index=False))
    print("\nSaved benchmark table to artifacts/tier0_tier1_cross_hospital.csv")
