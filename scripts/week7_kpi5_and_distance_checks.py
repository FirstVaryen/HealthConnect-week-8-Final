import os
import pandas as pd, numpy as np
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "healthconnect_appointments_prepared.csv")
df = pd.read_csv(P)
rng = np.random.default_rng(42)
print("OUTCOMES:", df.appointment_outcome.value_counts().to_dict())
df["canc"] = df.appointment_outcome.str.contains("Cancel", case=False)
df["ns"] = df.is_no_show.astype(bool)

def ratio_ci(d, B=2000):
    a = np.column_stack([d.canc.values, d.ns.values]); n = len(a); v = []
    for _ in range(B):
        s = a[rng.integers(0, n, n)]; v.append(s[:, 0].sum() / s[:, 1].sum())
    return d.canc.sum() / d.ns.sum(), *np.percentile(v, [2.5, 97.5])

print("\n=== PART A: KPI 5 cancellation-to-no-show ratio ===")
r = ratio_ci(df); print(f"Overall n={len(df)} ratio={r[0]:.3f} CI [{r[1]:.3f}, {r[2]:.3f}]")
for b in ["0-7 days", "8-14 days", "15-30 days", "31-45 days", "46-60 days"]:
    d = df[df.lead_time_band == b]; r = ratio_ci(d)
    print(f"{b:11s} n={len(d):5d} ratio={r[0]:.3f} CI [{r[1]:.3f}, {r[2]:.3f}]")

print("\n=== PART B: distance effect inside each lead-time band (distance known only) ===")
dk = df[df.distance_known].copy(); dk["far"] = dk.distance_to_clinic_km >= 15
print("Rows with known distance:", len(dk), "of", len(df))
print("Spearman corr distance vs lead days: %.3f" % dk.distance_to_clinic_km.rank().corr(dk.booking_lead_days.rank()))
for b in ["0-7 days", "8-14 days", "15-30 days", "31-45 days", "46-60 days"]:
    d = dk[dk.lead_time_band == b]; n1, n0 = d.far.sum(), (~d.far).sum()
    p1, p0 = d[d.far].ns.mean(), d[~d.far].ns.mean()
    se = np.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0); diff = p1 - p0
    print(f"{b:11s} <15km: {p0:.1%} (n={n0})  15+km: {p1:.1%} (n={n1})  diff={diff*100:+.1f}pp CI [{(diff-1.96*se)*100:+.1f}, {(diff+1.96*se)*100:+.1f}]")
d = dk; n1, n0 = d.far.sum(), (~d.far).sum(); p1, p0 = d[d.far].ns.mean(), d[~d.far].ns.mean()
print(f"ALL bands   <15km: {p0:.1%}  15+km: {p1:.1%}  diff={(p1-p0)*100:+.1f}pp")

print("\n=== PART C: Logistic Regression with vs without distance (5-fold CV ROC-AUC) ===")
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    dk["x"] = dk.booking_lead_days * dk.previous_no_shows
    base = ["booking_lead_days", "previous_no_shows", "x"]
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    for name, cols in [("without distance", base), ("with distance (km)", base + ["distance_to_clinic_km"]), ("with 15+km flag", base + ["far"])]:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        s = cross_val_score(m, dk[cols].astype(float), dk.ns, cv=cv, scoring="roc_auc")
        print(f"{name:20s} AUC {s.mean():.4f} +/- {s.std():.4f}")
except Exception as e:
    print("sklearn step failed:", repr(e))
