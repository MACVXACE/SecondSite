import joblib
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from data_prep import load_multi_hospital_data, CHECKUP_FEATURES, ClinicalCheckupEngineer

try:
    import lightgbm as lgb
    HAS_LGB = True
except (ImportError, OSError):
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGB = False


def build_checkup_preprocessor():
    cat_cols = ["sex", "cp", "fbs", "restecg"]
    num_cols = ["age", "trestbps", "chol", "is_hypertensive", "asymptomatic_cp", "age_x_sbp"]
    known_cats = [[0.0, 1.0], [1.0, 2.0, 3.0, 4.0], [0.0, 1.0], [0.0, 1.0, 2.0]]
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(categories=known_cats, drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])


def build_tier2_pipeline(random_state: int = 42) -> Pipeline:
    if HAS_LGB:
        clf = lgb.LGBMClassifier(
            n_estimators=110, learning_rate=0.03, max_depth=3, num_leaves=7,
            min_child_samples=15, subsample_for_bin=200, colsample_bytree=0.85,
            subsample=0.85, reg_alpha=0.5, reg_lambda=1.0, random_state=random_state, verbose=-1,
        )
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=110, learning_rate=0.03, max_depth=3, max_leaf_nodes=7,
            min_samples_leaf=15, l2_regularization=1.0, random_state=random_state,
        )
    return Pipeline([("eng", ClinicalCheckupEngineer()), ("prep", build_checkup_preprocessor()), ("clf", clf)])


def select_validation_thresholds(y_val: np.ndarray, oof_probs: np.ndarray, target_sens: float = 0.85):
    best_t, best_spec = 0.50, -1.0
    for t in np.linspace(0.20, 0.65, 91):
        preds = (oof_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if sens >= target_sens and spec > best_spec:
            best_spec, best_t = spec, float(round(t, 3))
    return float(round(max(0.18, best_t - 0.11), 3)), best_t, float(round(min(0.82, best_t + 0.13), 3))


def evaluate_with_abstention(label: str, y_true: np.ndarray, y_prob: np.ndarray, t_low: float, t_star: float, t_high: float) -> dict:
    y_pred = (y_prob >= t_star).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    unsure_mask = (y_prob >= t_low) & (y_prob <= t_high)
    return {
        "Cohort / Tier": label, "N": int(len(y_true)),
        "Accuracy (@t*)": round(accuracy_score(y_true, y_pred), 3),
        "Sensitivity (@t*)": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "Specificity (@t*)": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 3),
        "FN (@t*)": int(fn), "FP (@t*)": int(fp),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 3),
        "PR-AUC": round(average_precision_score(y_true, y_prob), 3),
        "Brier": round(brier_score_loss(y_true, y_prob), 3),
        "Unsure %": f"{unsure_mask.mean():.1%}",
        "Unflagged FN (Sent Home)": int(((y_true == 1) & (y_prob < t_low)).sum()),
        "Positives Caught by Unsure": int(((y_true == 1) & unsure_mask).sum()),
    }


def recalibrate_site_intercept(probs: np.ndarray, calib_y: np.ndarray, calib_probs: np.ndarray):
    eps = 1e-6
    logits_cal = logit(np.clip(calib_probs, eps, 1 - eps))
    best_delta, best_loss = 0.0, 1e9
    for delta in np.linspace(-2.5, 3.5, 241):
        loss = brier_score_loss(calib_y, expit(logits_cal + delta))
        if loss < best_loss:
            best_loss, best_delta = loss, float(delta)
    return expit(logit(np.clip(probs, eps, 1 - eps)) + best_delta), best_delta


if __name__ == "__main__":
    df = load_multi_hospital_data()
    cleveland = df[df["hospital"] == "Cleveland"].copy()
    train_df, test_df = train_test_split(cleveland, test_size=0.20, stratify=cleveland["target"], random_state=42)
    X_train, y_train = train_df[CHECKUP_FEATURES], train_df["target"].values
    X_test, y_test = test_df[CHECKUP_FEATURES], test_df["target"].values

    tier2_uncal = build_tier2_pipeline(random_state=42).fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tier3_cal = CalibratedClassifierCV(estimator=build_tier2_pipeline(random_state=42), method="sigmoid", cv=cv).fit(X_train, y_train)
    oof_probs = cross_val_predict(tier3_cal, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    t_low, t_star, t_high = select_validation_thresholds(y_train, oof_probs, target_sens=0.85)

    rows = [evaluate_with_abstention("Tier 3: Calibrated 7-Feat Model (Cleveland Test)", y_test, tier3_cal.predict_proba(X_test)[:, 1], t_low, t_star, t_high)]
    site_deltas = {"Cleveland (Primary)": 0.0}
    for hosp in ["Hungary", "Switzerland", "Long Beach VA"]:
        ext = df[df["hospital"] == hosp].copy()
        rows.append(evaluate_with_abstention(f"Unseen ({hosp}) — Zero-Shot", ext["target"].values, tier3_cal.predict_proba(ext[CHECKUP_FEATURES])[:, 1], t_low, t_star, t_high))
        cal_slice, eval_slice = train_test_split(ext, train_size=20, stratify=ext["target"], random_state=42)
        eval_recal, delta = recalibrate_site_intercept(
            tier3_cal.predict_proba(eval_slice[CHECKUP_FEATURES])[:, 1],
            cal_slice["target"].values,
            tier3_cal.predict_proba(cal_slice[CHECKUP_FEATURES])[:, 1],
        )
        site_deltas[f"{hosp} (20-Pt Recalibrated)"] = delta
        rows.append(evaluate_with_abstention(f"Unseen ({hosp} N={len(eval_slice)}) — 20-Pt Recalibrated", eval_slice["target"].values, eval_recal, t_low, t_star, t_high))

    pd.DataFrame(rows).to_csv("artifacts/tier3_calibrated_and_recalibrated.csv", index=False)
    joblib.dump({"tier2_uncal": tier2_uncal, "tier3_cal": tier3_cal, "t_low": t_low, "t_star": t_star, "t_high": t_high, "site_deltas": site_deltas}, "artifacts/secondsite_bundle.joblib")
    print("DONE: Saved tier3_calibrated_and_recalibrated.csv and secondsite_bundle.joblib")
