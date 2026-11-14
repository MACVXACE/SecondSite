import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit, expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from data_prep import load_multi_hospital_data, CHECKUP_FEATURES


class ClinicalCheckupEngineer(BaseEstimator, TransformerMixin):
    """Row-wise domain feature engineering using strictly routine checkup vitals (zero cross-row leakage)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X[CHECKUP_FEATURES].copy()
        df["is_hypertensive"] = (df["trestbps"] >= 140).astype(float)
        df["asymptomatic_cp"] = (df["cp"] == 4).astype(float)
        df["age_x_sbp"] = (df["age"] * df["trestbps"].fillna(130.0)) / 100.0
        return df


def build_checkup_preprocessor():
    cat_cols = ["sex", "cp", "fbs", "restecg"]
    num_cols = ["age", "trestbps", "chol", "is_hypertensive", "asymptomatic_cp", "age_x_sbp"]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def build_tier2_pipeline(random_state: int = 42) -> Pipeline:
    clf = lgb.LGBMClassifier(
        n_estimators=110,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=7,
        min_child_samples=15,
        subsample_for_bin=200,
        colsample_bytree=0.85,
        subsample=0.85,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=random_state,
        verbose=-1,
    )
    return Pipeline([
        ("eng", ClinicalCheckupEngineer()),
        ("prep", build_checkup_preprocessor()),
        ("clf", clf),
    ])


def select_validation_thresholds(y_val: np.ndarray, oof_probs: np.ndarray, target_sens: float = 0.85):
    """Select high-sensitivity triage threshold and uncertainty abstention band strictly on OOF validation data."""
    best_t = 0.50
    best_spec = -1.0

    for t in np.linspace(0.20, 0.65, 91):
        preds = (oof_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if sens >= target_sens and spec > best_spec:
            best_spec = spec
            best_t = float(round(t, 3))

    # Uncertainty gray zone around decision boundary: patients routed to "Order Exercise Stress Test"
    t_low = float(round(max(0.18, best_t - 0.11), 3))
    t_high = float(round(min(0.82, best_t + 0.13), 3))
    return t_low, best_t, t_high


def evaluate_with_abstention(label: str, y_true: np.ndarray, y_prob: np.ndarray, t_low: float, t_star: float, t_high: float) -> dict:
    # Binary evaluation at clinical threshold t_star
    y_pred = (y_prob >= t_star).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Three-zone triage evaluation (Low Risk | Unsure -> Stress Test | High Risk)
    unsure_mask = (y_prob >= t_low) & (y_prob <= t_high)
    definite_high = y_prob > t_high
    definite_low = y_prob < t_low

    # Missed cases that were confidently sent home as Low Risk (not caught by Unsure flag)
    unflagged_fn = int(((y_true == 1) & definite_low).sum())
    caught_by_unsure = int(((y_true == 1) & unsure_mask).sum())

    return {
        "Cohort / Tier": label,
        "N": int(len(y_true)),
        "Accuracy (@t*)": round(accuracy_score(y_true, y_pred), 3),
        "Sensitivity (@t*)": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "Specificity (@t*)": round(spec, 3),
        "FN (@t*)": int(fn),
        "FP (@t*)": int(fp),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 3),
        "PR-AUC": round(average_precision_score(y_true, y_prob), 3),
        "Brier": round(brier_score_loss(y_true, y_prob), 3),
        "Unsure %": f"{unsure_mask.mean():.1%}",
        "Unflagged FN (Sent Home)": unflagged_fn,
        "Positives Caught by Unsure": caught_by_unsure,
    }


def recalibrate_site_intercept(probs: np.ndarray, calib_y: np.ndarray, calib_probs: np.ndarray) -> np.ndarray:
    """Lightweight prior-shift / intercept recalibration using a small labeled site sample."""
    eps = 1e-6
    p_clip = np.clip(calib_probs, eps, 1 - eps)
    logits_cal = logit(p_clip)

    # Optimize scalar intercept shift delta so mean calibrated probability matches site prevalence
    best_delta = 0.0
    best_loss = 1e9
    for delta in np.linspace(-2.5, 3.5, 241):
        p_adj = expit(logits_cal + delta)
        loss = brier_score_loss(calib_y, p_adj)
        if loss < best_loss:
            best_loss = loss
            best_delta = float(delta)

    return expit(logit(np.clip(probs, eps, 1 - eps)) + best_delta), best_delta


if __name__ == "__main__":
    df = load_multi_hospital_data()
    cleveland = df[df["hospital"] == "Cleveland"].copy()

    # Exact same 80/20 stratified holdout split as baselines.py
    train_df, test_df = train_test_split(
        cleveland, test_size=0.20, stratify=cleveland["target"], random_state=42
    )

    X_train, y_train = train_df[CHECKUP_FEATURES], train_df["target"].values
    X_test, y_test = test_df[CHECKUP_FEATURES], test_df["target"].values

    # Fit uncalibrated Tier 2 pipeline for SHAP explanations
    tier2_uncal = build_tier2_pipeline(random_state=42)
    tier2_uncal.fit(X_train, y_train)

    # Fit Tier 3 Platt-calibrated model via 5-fold CV strictly on Cleveland train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tier3_cal = CalibratedClassifierCV(
        estimator=build_tier2_pipeline(random_state=42),
        method="sigmoid",
        cv=cv,
    )
    tier3_cal.fit(X_train, y_train)

    # Out-of-fold validation predictions on Cleveland train to choose clinical thresholds without test leakage
    oof_probs = cross_val_predict(
        tier3_cal, X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    t_low, t_star, t_high = select_validation_thresholds(y_train, oof_probs, target_sens=0.85)
    print(f"Validation-selected triage thresholds -> Low/Unsure: {t_low:.3f} | t*: {t_star:.3f} | Unsure/High: {t_high:.3f}\n")

    rows = []
    # Evaluate on locked Cleveland Test set
    test_probs = tier3_cal.predict_proba(X_test)[:, 1]
    rows.append(evaluate_with_abstention("Tier 3: Calibrated 7-Feat LGBM (Cleveland Test)", y_test, test_probs, t_low, t_star, t_high))

    # Evaluate on Unseen Hospitals + test 20-patient lightweight site recalibration
    site_deltas = {}
    for hosp in ["Hungary", "Switzerland", "Long Beach VA"]:
        ext = df[df["hospital"] == hosp].copy()
        ext_probs = tier3_cal.predict_proba(ext[CHECKUP_FEATURES])[:, 1]
        rows.append(evaluate_with_abstention(f"Unseen ({hosp}) — Zero-Shot", ext["target"].values, ext_probs, t_low, t_star, t_high))

        # 20-patient calibration slice vs remaining unseen site patients
        cal_slice, eval_slice = train_test_split(ext, train_size=20, stratify=ext["target"], random_state=42)
        cal_probs = tier3_cal.predict_proba(cal_slice[CHECKUP_FEATURES])[:, 1]
        eval_probs_raw = tier3_cal.predict_proba(eval_slice[CHECKUP_FEATURES])[:, 1]
        eval_probs_recal, delta = recalibrate_site_intercept(eval_probs_raw, cal_slice["target"].values, cal_probs)
        site_deltas[hosp] = delta
        rows.append(
            evaluate_with_abstention(
                f"Unseen ({hosp} N={len(eval_slice)}) — 20-Pt Recalibrated",
                eval_slice["target"].values,
                eval_probs_recal,
                t_low,
                t_star,
                t_high,
            )
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv("artifacts/tier3_calibrated_and_recalibrated.csv", index=False)
    print("=== TIER 3 CALIBRATED CHECKUP MODEL, ABSTENTION & SITE RECALIBRATION ===")
    print(summary_df.to_string(index=False))

    # Save frozen model bundle for evaluate.py and app.py
    bundle = {
        "tier2_uncal": tier2_uncal,
        "tier3_cal": tier3_cal,
        "t_low": t_low,
        "t_star": t_star,
        "t_high": t_high,
        "site_deltas": site_deltas,
        "X_train_bg": X_train.copy(),
    }
    joblib.dump(bundle, "artifacts/secondsite_bundle.joblib")
    print("\nSaved frozen model bundle to artifacts/secondsite_bundle.joblib")
