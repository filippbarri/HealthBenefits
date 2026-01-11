# WHO Physical Activity — Global EDA

This project explores the WHO Global Health Observatory indicator:
**Prevalence of insufficient physical activity among adults (18+), %**.

## What’s inside
- `notebooks/eda_clean.ipynb` — cleaned EDA notebook (answers Q1–Q6)
- `app.py` — Streamlit app (year/sex filters + plots)
- `data/insufficient_activity.csv` — dataset (place your CSV here)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this repo to GitHub.
2. On Streamlit Cloud: **New app** → select the repo → set main file to `app.py`.
3. Make sure `data/insufficient_activity.csv` is present in the repo (or replace the loader with a URL).

## Notes
- This is a descriptive analysis based on one dataset; no causal claims are made.
