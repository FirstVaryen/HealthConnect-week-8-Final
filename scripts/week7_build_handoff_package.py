import os, shutil
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
df = pd.read_csv(os.path.join(ROOT, "data", "processed", "healthconnect_appointments_prepared.csv"))
ns = df["is_no_show"].astype(bool)
lead, prior = df["booking_lead_days"], df["previous_no_shows"]
defs = [
    ("A validated DA definition", "booking_lead_days >= 31 AND previous_no_shows >= 2", (lead >= 31) & (prior >= 2), "Matches the Week 6 finding: about 4.7% of bookings, about 73% risk"),
    ("B as coded in train_improved.py", "booking_lead_days >= 46 AND previous_no_shows >= 2", (lead >= 46) & (prior >= 2), "Narrower and more extreme than the finding its code comment cites"),
    ("C left out by the coded version", "booking_lead_days 31 to 45 AND previous_no_shows >= 2", (lead >= 31) & (lead <= 45) & (prior >= 2), "Same stacking effect, silently excluded from the flag"),
]
rows = [[n, d, int(m.sum()), round(m.mean() * 100, 2), round(ns[m].mean() * 100, 1), note] for n, d, m, note in defs]
out = pd.DataFrame(rows, columns=["definition", "rule", "n_bookings", "share_of_5000_bookings_pct", "no_show_rate_pct", "note"])
dest = os.path.join(ROOT, "handoff", "ds_track_collaborator")
out.to_csv(os.path.join(dest, "double_risk_flag_segment_comparison.csv"), index=False)
for f in ["crosstrack_double_risk_flag_retest_evidence.csv", "crosstrack_double_risk_flag_lr_retest_evidence.csv"]:
    shutil.copy(os.path.join(ROOT, "reports", "cross_track_evidence", f), dest)
open(os.path.join(dest, "evidence", "PUT_SCREENSHOT_OF_SENT_MESSAGE_AND_REPLY_HERE.txt"), "w").write("Save a screenshot of the message you sent (with date) and any reply in this folder. That is the proof the information was provided.\n")
print(out.to_string(index=False))
print(sorted(os.listdir(dest)))
