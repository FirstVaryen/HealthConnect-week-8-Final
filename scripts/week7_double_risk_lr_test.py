import os
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "healthconnect_appointments_prepared.csv")
df = pd.read_csv(P); y = df.is_no_show.astype(int)
df["f46"] = ((df.booking_lead_days >= 46) & (df.previous_no_shows >= 2)).astype(int)
df["f31"] = ((df.booking_lead_days >= 31) & (df.previous_no_shows >= 2)).astype(int)
df["x"] = df.booking_lead_days * df.previous_no_shows
for f in ["f46", "f31"]:
    print(f"{f}: n={df[f].sum()} ({df[f].mean():.2%} of bookings), no-show rate {y[df[f]==1].mean():.1%}")
raw = ["booking_lead_days", "previous_no_shows"]
sets = {"A raw only": raw, "B raw + flag46": raw + ["f46"], "C raw + flag31": raw + ["f31"],
        "D raw + interaction": raw + ["x"], "E raw + interaction + flag46": raw + ["x", "f46"], "F raw + interaction + flag31": raw + ["x", "f31"]}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
res = {}
for k, cols in sets.items():
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    res[k] = cross_val_score(m, df[cols].astype(float), y, cv=cv, scoring="roc_auc")
    print(f"{k:30s} AUC {res[k].mean():.4f} +/- {res[k].std():.4f}")
for a, b in [("C raw + flag31", "B raw + flag46"), ("F raw + interaction + flag31", "E raw + interaction + flag46")]:
    d = res[a] - res[b]
    print(f"paired diff [{a}] minus [{b}]: {d.mean():+.4f} (sd {d.std():.4f}, wins {int((d>0).sum())}/{len(d)} folds)")
out = pd.DataFrame([{"model": k, "cv_roc_auc_mean": round(v.mean(), 4), "cv_roc_auc_std": round(v.std(), 4)} for k, v in res.items()])
out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "cross_track_evidence", "crosstrack_double_risk_flag_lr_retest_evidence.csv"), index=False)
