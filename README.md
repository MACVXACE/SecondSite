# SecondSite: Cross-Hospital Cardiovascular Triage with Routine Checkup Calibration and Uncertainty Abstention

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MACVXACE/SecondSite/blob/master/SecondSite_Full_Pipeline.ipynb)

**Track:** AI + Health  
**Team:** Varun Bajpai, Laziza Farrukhovna, Zainab Jamshed

> *This project is a prototype and is not medical advice.*

## Main Takeaway
A heart-disease risk model that looks strong inside one hospital drops sharply when deployed at unseen hospitals, and naive multi-hospital training leaks hospital identity through missing stress-test/imaging columns (`ca`, `thal` missing in 90-99% of non-Cleveland patients; `chol` 100% missing in Switzerland). **SecondSite** uses only **7 routine outpatient checkup measurements**, pairs calibrated probabilities with a validation-tuned **"Unsure - Order Exercise Stress Test"** zone (`25.0%-49.0%`), and audits demographic sensitivity gaps across sex and age.

## Key Results (Held-Out Test & Unseen Hospitals)
- **Tier 0 (Majority Baseline, Cleveland Test N=61):** Accuracy `0.541`, Sensitivity `0.000` (`28` missed cases).
- **Tier 1B (7-Feature Checkup Logistic Regression, Cleveland Test N=61):** ROC-AUC `0.868`, Sensitivity `0.714` (`8` missed cases), dropping to `0.755` ROC-AUC on Switzerland and `0.713` on Long Beach VA (`38` missed cases).
- **Tier 3 (Calibrated 7-Feature Checkup LightGBM + Abstention):**
  - **Cleveland Test (N=61):** ROC-AUC `0.854 [0.744, 0.952]`, Sensitivity `@t*=0.360` = `0.857`, and the Unsure Zone (`26.2%` of patients) catches 3 of the 4 remaining positives, leaving **only 1 unflagged patient sent home**.
  - **All Unseen Hospitals Combined (N=617):** ROC-AUC `0.839 [0.807, 0.872]`, Sensitivity `0.868 [0.834, 0.902]`, Specificity `0.652 [0.593, 0.709]`.
  - **Sex & Age Subgroup Parity (N=678 Held-Out):** Fixed thresholding yields `0.406` sensitivity for females (`19` FN) vs. `0.907` for males due to lower female baseline prevalence (`27.4%` vs. `65.2%`); SecondSite Unsure Zone rescues **12 of those 19 missed female cases** (`78.1%` total female capture).

## How to Run Locally
1. `pip install -r requirements.txt`
2. `python3 data_prep.py && python3 baselines.py && python3 models.py && python3 evaluate.py`
3. `python3 app.py`
