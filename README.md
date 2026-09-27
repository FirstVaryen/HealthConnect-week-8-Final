# HealthConnect Clinic - Week 8 (Data Analytics Track)

**Programme:** AnalystLab Africa Experience Lab
**Project:** Improving Patient Appointment Attendance and Healthcare Support Using Data and AI
**Track:** Data Analytics
**Prepared by:** Hillary Emmanuel
**Week 8 focus:** Final Integration, Presentation & Project Showcase

**Google Drive submission folder:** https://drive.google.com/drive/folders/1rps2d3KL30dUG4aORPZVC86csizsWc6t?usp=sharing

**Week 8 published social posts:** [LinkedIn](https://lnkd.in/p/egPBQVqk) | [X](https://x.com/Varyen01/status/2104319485528412592?s=20)

---

## What's in this repository

| Path | What it is |
|---|---|
| `reports/HealthConnect_Week8_Final_Analytics_Report.pdf` | Main Week 8 deliverable: transition write-up, final KPIs, validated findings, business insights, recommendations, limitations, executive summary |
| `reports/HealthConnect_Week8_Project_Summary.pdf` | The Week 8 Project Summary |
| `reports/Week8_Evidence_and_Supporting_Files_Manifest.pdf` | Packing list mapping each submission requirement to its file |
| `reports/cross_track_evidence/` | Evidence for the Week 8 HC-POD final integration activity |
| `outputs/dashboard/` | Final dashboard export and presentation slide deck (`HealthConnect_Week8_Presentation_Slides.pptx`) |
| `notebooks/HealthConnect_Week8_Final_Executed_Notebook_CarriedFromWeek7.ipynb` | Final executed notebook, carried forward from Week 7 unmodified |
| `scripts/` | Any supporting scripts used in Week 8 |
| `data/processed/` | Read-only copies of the prepared dataset carried forward from Week 7 |

---

## Reproducing the analysis

Any Python 3 environment with `pandas`, `numpy` and `scikit-learn` works (developed on Python 3.14).

```bash
pip install pandas numpy scikit-learn jupyter
cd notebooks
jupyter notebook
```

---

## Headline findings (carried from Week 7, finalised in Week 8)

- **Booking lead time is the strongest predictor of no-show risk.** Rate rises from 27.8% (0-7 days) to 67.7% (46-60 days), validated across all appointment types and age groups.
- **Double-risk segment: lead time 31+ days and 2+ prior no-shows.** 233 of 5,000 bookings, 73.0% no-show rate, bootstrap stable CI [67.4%, 79.0%].
- **Distance is a 15+ km threshold effect, not a gradient.** About +7 points in every lead-time band; real but modest.
- **Long lead-time bookings fail silently.** Cancellation-to-no-show ratio falls from ~0.20 (0-7 days) to ~0.08 (46-60 days).
- **Follow-up appointments are the highest-risk type.** 51.2% no-show rate overall, reaching 70% at 46-60 days lead time.

## Open items (carrying into Week 8)

- Distance-chart bar colour refinement: decided against. It is a styling only change with no effect on any KPI or finding, so it will not be applied to the live Power BI dashboard.
- DS track collaborator to merge the corrected double_risk_flag threshold (31+) into train_improved.py.
- Double-risk call pilot designed in Week 7 (report Section 9) — present design, not results.
- Individual 5-10 minute video presentation required.

See `reports/HealthConnect_Week8_Final_Analytics_Report.pdf` for the full write-up.
