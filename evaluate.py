import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from data_prep import load_multi_hospital_data, CHECKUP_FEATURES

CLEAN_FEATURE_NAMES = {
    "num__age": "Age (years)",
    "num__trestbps": "Resting BP (mmHg)",
    "num__chol": "Cholesterol (mg/dL)",
    "num__is_hypertensive": "Hypertensive (BP>=140)",
    "num__asymptomatic_cp": "Asymptomatic Chest Pain",
    "num__age_x_sbp": "Age x Resting BP",
    "cat__sex_1.0": "Sex: Male",
    "cat__cp_2.0": "Chest Pain: Atypical Angina",
    "cat__cp_3.0": "Chest Pain: Non-Anginal",
    "cat__cp_4.0": "Chest Pain: Asymptomatic",
    "cat__fbs_1.0": "Fasting Blood Sugar >120",
    "cat__restecg_1.0": "Resting ECG: ST-T Wave Abnorm.",
    "cat__restecg_2.0": "Resting ECG: LV Hypertrophy",
}


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {"acc": [], "sens": [], "spec": [], "roc": [], "pr": [], "brier": []}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        ypred = (yp >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, ypred, labels=[0, 1]).ravel()
        stats["acc"].append(accuracy_score(yt, ypred))
        stats["sens"].append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        stats["spec"].append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        stats["roc"].append(roc_auc_score(yt, yp))
        stats["pr"].append(average_precision_score(yt, yp))
        stats["brier"].append(brier_score_loss(yt, yp))

    def fmt(arr, val):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return f"{val:.3f} [{lo:.3f}, {hi:.3f}]"

    ypred0 = (y_prob >= threshold).astype(int)
    tn0, fp0, fn0, tp0 = confusion_matrix(y_true, ypred0, labels=[0, 1]).ravel()
    spec0 = tn0 / (tn0 + fp0) if (tn0 + fp0) > 0 else 0.0

    return {
        "N": n,
        "Accuracy (95% CI)": fmt(stats["acc"], accuracy_score(y_true, ypred0)),
        "Sensitivity (95% CI)": fmt(stats["sens"], recall_score(y_true, ypred0)),
        "Specificity (95% CI)": fmt(stats["spec"], spec0),
        "ROC-AUC (95% CI)": fmt(stats["roc"], roc_auc_score(y_true, y_prob)),
        "PR-AUC (95% CI)": fmt(stats["pr"], average_precision_score(y_true, y_prob)),
        "Brier (95% CI)": fmt(stats["brier"], brier_score_loss(y_true, y_prob)),
    }


def compute_shap_matrix(tier2_uncal, X_df: pd.DataFrame):
    eng = tier2_uncal.named_steps["eng"]
    prep = tier2_uncal.named_steps["prep"]
    clf = tier2_uncal.named_steps["clf"]

    X_trans = prep.transform(eng.transform(X_df))
    raw_names = prep.get_feature_names_out()
    clean_names = [CLEAN_FEATURE_NAMES.get(n, n) for n in raw_names]

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_trans)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv, X_trans, clean_names


if __name__ == "__main__":
    df = load_multi_hospital_data()
    bundle = joblib.load("artifacts/secondsite_bundle.joblib")
    tier2_uncal = bundle["tier2_uncal"]
    tier3_cal = bundle["tier3_cal"]
    t_low, t_star, t_high = bundle["t_low"], bundle["t_star"], bundle["t_high"]

    cleveland = df[df["hospital"] == "Cleveland"].copy()
    train_df, test_df = train_test_split(
        cleveland, test_size=0.20, stratify=cleveland["target"], random_state=42
    )

    # 1. Bootstrap 95% CIs across Cleveland Test & Unseen Hospitals
    ci_rows = []
    for name, subset in [
        ("Cleveland Test (Holdout)", test_df),
        ("Hungary (Unseen)", df[df["hospital"] == "Hungary"]),
        ("Switzerland (Unseen)", df[df["hospital"] == "Switzerland"]),
        ("Long Beach VA (Unseen)", df[df["hospital"] == "Long Beach VA"]),
        ("All Unseen Hospitals Combined", df[df["hospital"] != "Cleveland"]),
    ]:
        probs = tier3_cal.predict_proba(subset[CHECKUP_FEATURES])[:, 1]
        row = {"Cohort": name}
        row.update(bootstrap_ci(subset["target"].values, probs, threshold=t_star))
        ci_rows.append(row)

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv("artifacts/bootstrap_ci_summary.csv", index=False)
    print("=== HELD-OUT PERFORMANCE WITH 95% BOOTSTRAP CONFIDENCE INTERVALS ===")
    print(ci_df.to_string(index=False))

    # 2. Sex & Age Subgroup Fairness Audit (on all held-out patients: Cleveland Test + 3 Unseen Hospitals)
    held_out = pd.concat([test_df, df[df["hospital"] != "Cleveland"]], ignore_index=True)
    held_out["prob"] = tier3_cal.predict_proba(held_out[CHECKUP_FEATURES])[:, 1]
    held_out["pred"] = (held_out["prob"] >= t_star).astype(int)
    held_out["unsure"] = (held_out["prob"] >= t_low) & (held_out["prob"] <= t_high)

    subgroups = [
        ("Sex: Female (sex=0)", held_out["sex"] == 0),
        ("Sex: Male (sex=1)", held_out["sex"] == 1),
        ("Age: < 55 years", held_out["age"] < 55),
        ("Age: >= 55 years", held_out["age"] >= 55),
    ]

    fair_rows = []
    for label, mask in subgroups:
        sub = held_out[mask]
        tn, fp, fn, tp = confusion_matrix(sub["target"], sub["pred"], labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fair_rows.append({
            "Subgroup": label,
            "N": int(len(sub)),
            "Prevalence": round(float(sub["target"].mean()), 3),
            "Sensitivity (Recall)": round(sens, 3),
            "Specificity": round(spec, 3),
            "Missed Cases (FN)": int(fn),
            "False Alarms (FP)": int(fp),
            "ROC-AUC": round(roc_auc_score(sub["target"], sub["prob"]), 3),
            "Caught by Unsure Zone": int(((sub["target"] == 1) & sub["unsure"] & (sub["pred"] == 0)).sum()),
        })

    fair_df = pd.DataFrame(fair_rows)
    fair_df.to_csv("artifacts/subgroup_fairness_audit.csv", index=False)
    print("\n=== SEX & AGE SUBGROUP FAIRNESS AUDIT (ALL 678 HELD-OUT PATIENTS) ===")
    print(fair_df.to_string(index=False))

    # 3. Extract Concrete Failure Modes (False Negatives & False Positives on Cleveland Test)
    test_eval = test_df.copy().reset_index(drop=True)
    test_eval["prob"] = tier3_cal.predict_proba(test_eval[CHECKUP_FEATURES])[:, 1]
    test_eval["pred"] = (test_eval["prob"] >= t_star).astype(int)
    sv_test, _, clean_names = compute_shap_matrix(tier2_uncal, test_eval[CHECKUP_FEATURES])

    print("\n=== CONCRETE HELD-OUT FAILURE CASES (FOR WRITE-UP & DEMO) ===")
    fn_cases = test_eval[(test_eval["target"] == 1) & (test_eval["pred"] == 0)]
    fp_cases = test_eval[(test_eval["target"] == 0) & (test_eval["pred"] == 1)]
    print(f"Cleveland Test Missed Cases (FN): {len(fn_cases)} | False Alarms (FP): {len(fp_cases)}")

    for case_type, subset_cases in [("FALSE NEGATIVE (Missed Disease)", fn_cases), ("FALSE POSITIVE (False Alarm)", fp_cases)]:
        if len(subset_cases) > 0:
            idx = subset_cases.index[0]
            row = test_eval.loc[idx]
            shap_pairs = sorted(zip(clean_names, sv_test[idx]), key=lambda x: abs(x[1]), reverse=True)[:3]
            drivers = ", ".join([f"{k} ({v:+.2f})" for k, v in shap_pairs])
            print(
                f"-> {case_type} [Test Row {idx}]: age={row['age']:.0f}, sex={int(row['sex'])}, "
                f"cp={int(row['cp'])}, trestbps={row['trestbps']:.0f}, chol={row['chol']:.0f}, "
                f"fbs={int(row['fbs'])}, restecg={int(row['restecg'])} | Calibrated Risk={row['prob']:.1%} | Top SHAP: {drivers}"
            )

    # 4. Generate 3 Publication Plots
    # Plot 1: Cross-Hospital ROC-AUC & Sensitivity Drop
    base_df = pd.read_csv("artifacts/tier0_tier1_cross_hospital.csv")
    t3_df = pd.read_csv("artifacts/tier3_calibrated_and_recalibrated.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    cohorts = ["Cleveland Test", "Hungary", "Switzerland", "Long Beach VA"]
    t1_rocs = [
        base_df.loc[base_df["Cohort / Model"].str.contains("Tier 1B"), "ROC-AUC"].values[0],
        base_df.loc[base_df["Cohort / Model"].str.contains("Hungary"), "ROC-AUC"].values[0],
        base_df.loc[base_df["Cohort / Model"].str.contains("Switzerland"), "ROC-AUC"].values[0],
        base_df.loc[base_df["Cohort / Model"].str.contains("Long Beach"), "ROC-AUC"].values[0],
    ]
    t3_rocs = [
        t3_df.loc[t3_df["Cohort / Tier"].str.contains("Cleveland Test"), "ROC-AUC"].values[0],
        t3_df.loc[t3_df["Cohort / Tier"].str.contains("Hungary.*Zero-Shot"), "ROC-AUC"].values[0],
        t3_df.loc[t3_df["Cohort / Tier"].str.contains("Switzerland.*Zero-Shot"), "ROC-AUC"].values[0],
        t3_df.loc[t3_df["Cohort / Tier"].str.contains("Long Beach.*Zero-Shot"), "ROC-AUC"].values[0],
    ]
    x = np.arange(len(cohorts))
    ax.bar(x - 0.18, t1_rocs, width=0.35, label="Tier 1B: 7-Feat Logistic Regression", color="#64748b")
    ax.bar(x + 0.18, t3_rocs, width=0.35, label="Tier 3: Calibrated Checkup LightGBM", color="#0284c7")
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Cross-Hospital Generalization (Trained on Cleveland Only)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("artifacts/fig1_cross_hospital_drop.png", dpi=200)
    plt.close()

    # Plot 2: Sex & Age Subgroup Parity
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    sx = np.arange(len(fair_df))
    ax.bar(sx - 0.18, fair_df["Sensitivity (Recall)"], width=0.35, label="Sensitivity (Recall)", color="#dc2626")
    ax.bar(sx + 0.18, fair_df["Specificity"], width=0.35, label="Specificity", color="#2563eb")
    ax.set_xticks(sx)
    ax.set_xticklabels(fair_df["Subgroup"], rotation=12)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Sex & Age Subgroup Fairness Audit (All 678 Held-Out Patients)")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("artifacts/fig2_subgroup_fairness.png", dpi=200)
    plt.close()

    # Plot 3: Global SHAP Importance
    mean_abs_shap = np.abs(sv_test).mean(axis=0)
    order = np.argsort(mean_abs_shap)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.barh([clean_names[i] for i in order], mean_abs_shap[order], color="#0f766e")
    ax.set_xlabel("Mean |SHAP Value| (Impact on Log-Odds)")
    ax.set_title("SecondSite Checkup-Only Model: Global Feature Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("artifacts/fig3_shap_summary.png", dpi=200)
    plt.close()

    print("\nSaved 3 publication plots to artifacts/fig1_cross_hospital_drop.png, fig2_subgroup_fairness.png, fig3_shap_summary.png")
