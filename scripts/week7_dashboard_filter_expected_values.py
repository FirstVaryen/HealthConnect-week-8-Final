import os
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "healthconnect_appointments_prepared.csv"))
df["canc"] = df["appointment_outcome"].str.contains("Cancel", case=False)
df["ns"] = df["is_no_show"].astype(bool)

selections = [("(none)", "(none)", df)]
for f in ["appointment_type", "age_group", "reminder_sent"]:
    for v in sorted(df[f].dropna().unique()):
        selections.append((f, v, df[df[f] == v]))
priority = {("(none)", "(none)"): "T0", ("appointment_type", "Follow-up"): "T1", ("age_group", "65+"): "T2", ("reminder_sent", "Yes"): "T3"}

rows, other = [], 0
for f, v, d in selections:
    tid = priority.get((f, v))
    if tid is None:
        other += 1; tid = "X%02d" % other
    def add(visual, group, n, val): rows.append([tid, f, v, visual, group, n, val, "", ""])
    add("card: Total Appointments", "all", len(d), len(d))
    add("card: No-Show Count", "all", len(d), int(d["ns"].sum()))
    add("card: Cancellation Count", "all", len(d), int(d["canc"].sum()))
    add("card: No-Show Rate (%)", "all", len(d), round(d["ns"].mean() * 100, 1))
    for visual, col in [("KPI 1 no-show rate % by prior_no_show_bracket", "prior_no_show_bracket"), ("KPI 2 no-show rate % by lead_time_band", "lead_time_band"),
                        ("KPI 3 no-show rate % by distance_band", "distance_band"), ("KPI 4 no-show rate % by reminder_sent", "reminder_sent")]:
        for g, dd in d.groupby(col):
            add(visual, str(g), len(dd), round(dd["ns"].mean() * 100, 1))
    for g, dd in d.groupby("lead_time_band"):
        ns = int(dd["ns"].sum())
        add("KPI 5 Cancel to NoShow Ratio by lead_time_band", str(g), len(dd), round(dd["canc"].sum() / ns, 3) if ns else "n/a")
out = pd.DataFrame(rows, columns=["test_id", "slicer_field", "slicer_value", "visual", "group", "n_in_group", "expected_value", "actual_in_power_bi", "match_yes_no"])
p = os.path.join(HERE, "..", "reports", "test_evidence", "dashboard_filter_expected_values.csv")
out.to_csv(p, index=False)
print("rows:", len(out), "| selections:", len(selections))
print(out[out.test_id.isin(["T0", "T1", "T2", "T3"]) & out.visual.str.startswith("card")].drop(columns=["actual_in_power_bi", "match_yes_no", "group", "n_in_group"]).to_string(index=False))
