# SecondSite: Cross-Hospital Cardiovascular Triage with Routine Checkup Calibration and Uncertainty Abstention

**Track:** AI + Health  
**Team:** Varun Bajpai, Laziza Farrukhovna, Zainab Jamshed

> *This project is a prototype and is not medical advice.*

## The Problem & Our Scientific Proof
Heart-disease risk models trained on the standard 4-hospital UCI dataset often look highly accurate in tutorials. However, we discovered a massive data leak: stress-test missingness and physiological zero-coding perfectly leak hospital identity. 

**Our Leak Probe Result:** A model using *only* missing-value flags (with no actual medical vitals) can guess the hospital with **84.5% accuracy** and predict heart disease with a **0.745 ROC-AUC**. Multi-hospital training on this data forces models to cheat by learning hospital artifacts.

## The SecondSite Architecture
To build an honest clinical tool for resource-constrained outpatient clinics, SecondSite:
1. **Uses only 7 routine checkup measurements** (no upfront stress tests or imaging).
2. **Trains strictly on one complete hospital (Cleveland)** to completely isolate the model from missing-value leakage.
3. **Calibrates probabilities** using HistGradientBoosting and `CalibratedClassifierCV`.
4. **Implements an Uncertainty Abstention Band** derived from out-of-fold validation, dynamically separating patients into *Routine Follow-up*, *Unsure - Order Stress Test*, and *Refer for Cardiology*.

## Key Results (Held-Out & Unseen Generalization)
* **Tier 0 & Tier 1 (Baselines):** The standard Logistic Regression achieves 0.821 sensitivity on Cleveland, but drops aggressively to **0.453 sensitivity** when tested on the unseen Long Beach VA hospital. 
* **Tier 3 (Calibrated SecondSite - 3 Unseen Hospitals Pooled, N=615):** 
  * **ROC-AUC:** 0.829
  * **Referral Precision:** 85.3% (Over 4 out of 5 patients in our 'Urgent Referral' band actually had heart disease).

## Fixing the Demographic Sensitivity Gap
Under a shared clinical threshold, our fairness audit revealed a dangerous disparity: female sensitivity on unseen hospitals was only **48.0%** (13 missed cases) compared to **95.3%** for males, due to lower baseline female prevalence in the training set.

**The Fix:** We implemented sex-stratified rule-out lines optimized strictly on Cleveland out-of-fold predictions. 
* **Result:** Female sensitivity on unseen hospitals jumped to **84.0%** (only 4 missed cases).
* **Gap Closed:** The sensitivity gap between men and women dropped from `0.473` down to just `0.082`.

## Reproducing & Running the App
Step 1: pip install -r requirements.txt
Step 2: python3 data_prep.py && python3 baselines.py && python3 models.py
Step 3: python3 app.py
