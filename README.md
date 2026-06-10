# LiftLab

> A cloud-native A/B testing platform built end-to-end on real GA4 e-commerce data.

LiftLab simulates and analyzes online controlled experiments the way a production platform at Microsoft, Netflix, or LinkedIn would — user-level inference, CUPED variance reduction, Bayesian decision theory, heterogeneous-effect segmentation, policy guardrails, and an integrated ship/hold/reject/investigate recommendation. The pipeline runs on the Google Merchandise Store GA4 BigQuery export with synthetic treatment effects injected against a known ground truth.


---

## Stack

`Python` · `FastAPI` · `Streamlit` · `Plotly` · `NumPy` · `SciPy` · `Pandas` · `Pydantic` · `Docker` · `GCP / BigQuery / Cloud Run`

---

## The headline result

On the default experiment (`Checkout Flow Redesign`, 195,884 users, 50/50 stratified by device, Dec 2020 – Jan 2021):

| Layer | Result |
|---|---|
| **SRM check** | PASS (chi² = 0.0007, p = 0.98) |
| **Conversion rate** | +6.69%, p = 0.07 (frequentist HOLD), P(T>C) = 0.96 (Bayesian SHIP) |
| **Revenue per user** | +18.22%, p = 0.007, P(T>C) = 0.997 |
| **Guardrails** | bounce PASS, depth PASS, **volatility FAIL (+21% vs 15% limit)** |
| **Final verdict** | **HOLD** — topline is great, but revenue volatility breached policy |

This is a common scenario the platform is built to surface: a "the headline numbers look good but a guardrail caught a real risk" call that pure NHST would miss

---

## Quick start

### Docker (recommended)

```bash
docker-compose up
```

- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

The first startup runs the full clean → assign → simulate pipeline once and caches the resulting DataFrame in memory. After that the endpoints are instant; heavier ones (segmentation, decision) take a few seconds because they recompute on demand.

### Local

```bash
pip install -r requirements.txt

# API
uvicorn app.api.main:app --reload

# Dashboard (new terminal)
streamlit run app/dashboard/Home.py
```

### Run the stats engine directly

```bash
python src/frequentist.py     # user-level z-test / Welch's t + SRM
python src/bayesian.py        # Beta-Binomial / Normal posterior + expected loss
python src/cuped.py           # variance reduction
python src/segmentation.py    # heterogeneous effects
python src/guardrails.py      # policy thresholds
python src/power.py           # MDE + pre-experiment sample sizing
python src/recommendation.py  # final SHIP / HOLD / REJECT / INVESTIGATE
```

### Tests

```bash
pytest tests/ -v
```

### Notebooks

- `notebooks/01_eda.ipynb` — narrative end-to-end walkthrough on the default experiment
- `notebooks/02_alternate_experiments.ipynb` — runs two additional experiments through the same pipeline to demonstrate the **REJECT** and **SHIP** branches of the decision tree (the default lands on **HOLD**)

---

## Known limitations

- **CUPED variance reduction is small (0–2%)** on this dataset because most users don't have a real pre-period. The implementation is correct; the data is the limit. Production-scale longitudinal data would produce textbook 20–50% reductions.
- **No multiple-testing correction** across the secondary metrics — significance flags should be read as suggestive, not definitive at scale.

---
