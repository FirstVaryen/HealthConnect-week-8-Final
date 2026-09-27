"""
Week 7 cross-track retest on the EXACT Week 6 candidate model.

Candidate (week 6 - data science track/outputs/models/candidate_model.json):
Logistic Regression, restricted DA-approved feature set, StandardScaler,
class_weight balanced, max_iter 2000, 5-fold stratified CV ROC-AUC.

This script imports the Week 6 code itself (build_restricted_features and
build_model_frame), so the features cannot drift from the real candidate.
Only the double_risk_flag lead-time threshold is changed: 46 (as built in
Week 6) versus 31 (the value that reproduces the DA finding).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DS = Path(r"C:\Users\hilla\OneDrive\Desktop\AnalystLab\week 6 - data science track")
sys.path.insert(0, str(DS / "scripts"))
from prep_features import build_model_frame, load_raw  # noqa: E402
from robustness_check import build_restricted_features  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "cross_track_evidence"
work = build_model_frame(load_raw())
y = work["is_no_show"].astype(int)
print("Rows used (Attended + No-Show only):", len(work))


def features(threshold):
    X = build_restricted_features(work)  # Week 6 code, flag built at 46
    if threshold is None:
        return X.drop(columns=["double_risk_flag"])
    X["double_risk_flag"] = ((work["booking_lead_days"] >= threshold) & (work["previous_no_shows"] >= 2)).astype(int)
    return X


def model():
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    return Pipeline([("scaler", StandardScaler()), ("model", lr)])

variants = {"flag46 (Week 6 as built)": 46, "flag31 (DA finding)": 31, "no flag": None}
rows, folds = [], {}
# Part 1: Week 6 protocol exactly (single 5-fold, random_state 42). The flag46 row should
# reproduce candidate_model.json (0.6819) if this script really is the same model.
cv1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Part 2: 5x5 repeated CV, so paired differences are not one lucky split.
cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
for label, thr in variants.items():
    X = features(thr)
    s1 = cross_val_score(model(), X, y, cv=cv1, scoring="roc_auc")
    s2 = cross_val_score(model(), X, y, cv=cv2, scoring="roc_auc")
    folds[label] = s2
    rows.append({"variant": label, "n_features": X.shape[1],
                 "week6_protocol_auc_mean": round(s1.mean(), 4), "week6_protocol_auc_std": round(s1.std(), 4),
                 "repeated_cv_auc_mean": round(s2.mean(), 4), "repeated_cv_auc_std": round(s2.std(), 4)})
    print(f"{label:26s} features={X.shape[1]}  W6 protocol {s1.mean():.4f} +/- {s1.std():.4f}   5x5 CV {s2.mean():.4f} +/- {s2.std():.4f}")

a, b = folds["flag31 (DA finding)"], folds["flag46 (Week 6 as built)"]
d = a - b
print(f"paired diff flag31 minus flag46: {d.mean():+.4f} (sd {d.std():.4f}, flag31 wins {int((d > 0).sum())}/{len(d)} folds)")
print("Week 6 candidate_model.json reports CV AUC 0.6819 +/- 0.0120 for comparison")
for thr in (46, 31):
    f = (work["booking_lead_days"] >= thr) & (work["previous_no_shows"] >= 2)
    print(f"flag{thr}: n={int(f.sum())} ({f.mean():.2%} of modelled rows), no-show rate {y[f].mean():.1%}")
pd.DataFrame(rows).to_csv(OUT / "crosstrack_double_risk_flag_exact_candidate_retest_evidence.csv", index=False)
print("Saved evidence csv to", OUT)
