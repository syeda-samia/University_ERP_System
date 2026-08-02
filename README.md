# University ERP — Predictive Analytics System

A predictive analytics layer for a university ERP: six prediction modules
covering student risk, dropout, fee default, GPA, enrollment forecasting,
and course recommendations — built as a portfolio piece to demonstrate a
production-grade ML workflow: real feature engineering from raw records,
cross-validated model comparison, an Excel/CSV import pipeline any
university can plug their own data into, and honest reporting of what's
real versus simulated in the training data.

Two live surfaces, both from this one repo:
- **Production dashboard:** https://technify-five.vercel.app (FastAPI + plain HTML/JS, unchanged from before this upgrade)
- **Portfolio demo:** `streamlit_app.py` (this upgrade) — Excel/CSV upload, sample dataset, charts, model performance

---

## The 6 modules

| Module | What it predicts | Inputs |
|---|---|---|
| Student Risk | Chance a student fails this term | CGPA, attendance %, assignment completion rate, backlogs |
| Dropout | Dropout risk | Attendance %, CGPA, fees paid, LMS activity |
| Fee Default | Chance of defaulting on tuition | Gender, department, year level, status, city, scholarship, CGPA |
| GPA Prediction | Next-term GPA | Attendance %, quiz/assignment/exam averages, CGPA |
| Recommendation Engine | Which academic area needs attention | Quiz, assignment, exam, and lab averages |
| Enrollment Forecasting | Future enrollment counts | Historical enrollment dates |

---

## Excel/CSV import — bring your own data

Any university can use this system with their own student data through
**one flat template** — download it from the Streamlit app ("Download
sample template") or use `sample_data/sample_students_template.xlsx`.

### Required column

| Column | Description |
|---|---|
| `StudentID` | Unique student identifier. Always required. |

### Optional columns (each module runs only if its columns are present)

| Column | Type | Used by |
|---|---|---|
| `CGPA` | 0.0–4.0 | Student Risk, Fee Default, GPA |
| `AttendancePercentage` | 0–100 | Student Risk, Dropout, GPA |
| `AssignmentsSubmitted` | count | Student Risk |
| `AssignmentsTotal` | count | Student Risk |
| `Backlogs` | count of failed/backlog courses | Student Risk |
| `QuizAvg` | 0–100 | GPA, Recommendations |
| `AssignmentAvg` | 0–100 | GPA, Recommendations |
| `ExamAvg` | 0–100 | GPA, Recommendations |
| `LabAvg` | 0–100 | Recommendations |
| `FeesPaid` | 0 or 1 | Dropout |
| `LMSActivity` | 0–100 engagement score | Dropout |
| `Gender` | Male / Female | Fee Default |
| `YearLevel` | Freshman / Sophomore / Junior / Senior | Fee Default |
| `Status` | Active / Inactive / Graduated | Fee Default |
| `Scholarship` | Yes / No | Fee Default |
| `Department` | any value | Fee Default |
| `City` | any value | Fee Default |
| `EnrollmentDate` | YYYY-MM-DD | Enrollment Forecasting |

**Validation behavior:** missing the `StudentID` column, duplicate/blank
IDs, or a file with no usable columns at all → upload rejected with a clear
error. Everything else degrades gracefully: out-of-range numeric values are
clipped and counted (not rejected), unrecognized categories (e.g. an
unfamiliar department name) fall back to the most common training
category, and a module whose required columns are missing is simply
skipped — the rest still run. `Department`/`City` accept *any* value, since
every university's own vocabulary is different.

---

## Model methodology

### Data

`models/university_dataset.xlsx` — 1,000 students, 10,000 attendance
records, 5,000 exam records (Midterm/Final/Quiz/Assignment/Lab Exam) from a
demo university. Per-student features are aggregated from the raw
Attendance and ExamRecords sheets (864 students have enough records to
featurize; 136 with no attendance/exam history are excluded).

### What's real vs. simulated (read this before trusting the numbers)

The original repo shipped 4 pre-trained `.pkl` files with **no training
script and no accompanying training data** — the features they expected
(`fees_paid`, `lms_activity`, `PreviousGPA`, `QuizScores`, etc.) don't exist
as columns anywhere in the included dataset. They were real, fitted models
(verified: correct coefficients, populated trees, not the "empty model"
failure this upgrade was asked to check for) but unreproducible black
boxes. This upgrade replaces them with models retrained from the actual
dataset in this repo, with the provenance of every feature and label
stated plainly:

**Real, derived directly from the data:** CGPA, Gender, Department, Year
Level, Status, City, Scholarship, EnrollmentDate (Students sheet);
AttendancePercentage (Attendance sheet); QuizAvg/AssignmentAvg/ExamAvg/
LabAvg/Backlogs/AssignmentsSubmitted/Total (aggregated from ExamRecords by
exam type). The Student Risk label (CGPA < 2.0 or ExamAvg < 50) is
likewise real-derived.

**Simulated, and documented as such in `src/dataset.py`:**
- `FeesPaid` and `LMSActivity` — the dataset has no fee-transaction table
  or LMS export, so these are simulated features. A real deployment
  replaces them with actual fee-ledger / LMS data via the Excel template's
  own `FeesPaid`/`LMSActivity` columns.
- The **Dropout label** was originally going to be `Status == 'Inactive'`
  (a real column) — until checking the correlation between `Status` and
  attendance/CGPA/fees/LMS activity turned up **|r| < 0.07 for all four**.
  `Status` appears to have been assigned independently of those columns
  when the dataset was generated, so no classifier can legitimately learn
  "dropout" from them using that label — training on it produced a model
  with **F1 = 0** (precision and recall both zero), the exact
  "empty/undertrained model" failure this upgrade was asked to fix.
  Rather than ship that, the dropout label is a documented, simulated
  function of the same four risk factors (low attendance + low CGPA +
  unpaid fees + low LMS activity → higher risk).
- **Fee Default label** — no fee-transaction data exists at all, so this
  is a documented rule (low CGPA + no scholarship raises default risk,
  plus noise).
- **GPA regression target** — only one CGPA snapshot exists per student
  (no multi-semester history), so the "next-term GPA" target is a
  documented formula built from real performance indicators (attendance,
  quiz/assignment/exam averages) plus noise, not an observed future value.

A real deployment should retrain Dropout and Fee Default on actual
historical outcomes as soon as they exist, and retrain GPA once
multi-semester history is available.

### Model comparison

Each classification module compared Logistic Regression vs. Random Forest
with 5-fold cross-validation (scoring F1); the GPA module compared Linear
Regression vs. Random Forest (scoring R²). The winner was tuned with
`RandomizedSearchCV` (15 iterations, 5-fold) when Random Forest won — three
of four modules picked Random Forest, one (GPA) picked plain Linear
Regression outright.

---

## Performance metrics (held-out test set, never used in training or tuning)

| Module | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Student Risk | Random Forest | 0.869 | 1.000 | 0.553 | 0.712 |
| Dropout | Random Forest | 0.854 | 0.308 | 0.286 | 0.296 |
| Fee Default | Random Forest | 0.677 | 0.653 | 0.561 | 0.604 |

| Module | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| GPA Prediction | Linear Regression | 0.888 | 0.142 | 0.112 |

Enrollment forecast: linear trend on real historical counts, **R² = 0.61**
(2024's partial-year count excluded from the fit — data only runs through
September). Forecast: 2024→230, 2025→256, 2026→282, 2027→309, 2028→335
(vs. 862 actual full-year enrollments from 2019-2023).

**Honest read on these numbers:** Student Risk is strong (F1=0.71,
precision 1.0 — when it flags a student, it's right, though it misses
~45% of at-risk students). GPA prediction is strong (R²=0.89). Fee Default
is moderate (F1=0.60) — expected, since its label is simulated from only
two signals (CGPA, scholarship) plus noise. **Dropout is the weakest
module by a wide margin** (F1=0.30) — its simulated label is intentionally
noisier (to avoid a trivially learnable rule), and it's the module most in
need of real historical outcomes before any real-world use. Feature
importance across modules consistently puts CGPA and attendance at the
top, which matches domain intuition.

---

## Model limitations

- **Small dataset.** 864 students after cleaning is small for 4 separate
  ML models; expect metric variance across retrains with different seeds.
- **Two labels and two features are simulated, not observed** (see above)
  — this is a demo/portfolio system, not validated against real student
  outcomes. Don't deploy the Dropout or Fee Default modules to make real
  decisions about real students without retraining on real outcomes first.
- **Enrollment forecast is a simple linear trend**, not Prophet/ARIMA —
  deliberately, to avoid a heavy native dependency (see below) — and is
  fit on only 5-6 years of historical data. Treat it as directional, not precise.
- **Recommendation Engine works on aggregate scores, not per-course
  detail.** The original repo's version flagged specific weak *courses*
  from granular per-course exam records; this generalized version flags
  weak *academic areas* (quizzes/assignments/exams/labs) from the same
  aggregate columns every other module uses, so one template works for
  every module and any university's data — at the cost of losing
  per-course granularity.
- **Excel template is comprehensive but a lot of columns.** A university
  missing some (e.g. no LMS) still gets the modules that don't need them,
  but won't get all 6 out of the box.

## Why no XGBoost/LightGBM/Prophet

Random Forest / Logistic / Linear Regression from scikit-learn (already a
dependency) cover the same "ensemble vs. linear" comparison the brief
asked for without adding compiled native dependencies. A previous project
in this portfolio lost 25+ minutes to an XGBoost install hanging on this
machine's Python version with no prebuilt wheel; Prophet carries a similar
risk (Stan/cmdstanpy). Swapping either in is a drop-in change to
`src/train.py`'s candidate dictionaries if the deployment target has
stable wheels for them.

---

## Project structure

```
├── api/index.py                 # EXISTING production FastAPI backend (untouched)
├── index.html                   # EXISTING production dashboard (untouched)
├── models/
│   ├── *.pkl                    # EXISTING models used by api/index.py (untouched)
│   ├── university_dataset.xlsx  # Source dataset
│   └── erp_models.pkl           # NEW: consolidated bundle for streamlit_app.py
├── src/
│   ├── schema.py                # Canonical column schema + per-module requirements
│   ├── dataset.py                # Feature engineering, label construction, enrollment trend
│   ├── encoding.py               # Robust categorical encoding (unseen -> fallback)
│   ├── evaluate.py               # Metrics, cross-validation, feature importance
│   ├── train.py                  # Trains + tunes + saves all 4 ML modules
│   ├── validation.py             # Excel/CSV upload reading + validation
│   └── predict.py                # Runs all 6 modules on validated data
├── tests/                        # pytest: dataset, validation, prediction
├── sample_data/                  # Downloadable template + demo dataset
├── streamlit_app.py               # NEW: portfolio demo (separate from production dashboard)
├── requirements.txt               # Shared by both deployments (Vercel + Streamlit Cloud)
└── requirements-dev.txt           # + pytest
```

---

## How to run

### Streamlit demo (this upgrade)

```bash
pip install -r requirements.txt
python -m src.train          # retrains all 4 ML modules, writes models/erp_models.pkl
streamlit run streamlit_app.py
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Existing production dashboard (unchanged)

```bash
npm i -g vercel
vercel dev
```
See the "Running it yourself" section further down for the original setup — nothing about that deployment changed.

---

## Future improvements

- Retrain Dropout and Fee Default on real historical outcomes as soon as a
  university using this system can supply them.
- Add multi-semester GPA history so the GPA module predicts an observed
  next-term GPA rather than a formula-derived target.
- Swap in XGBoost/LightGBM and Prophet once deployed somewhere with stable
  prebuilt wheels for them, and re-run the same comparison harness.
- Restore per-course granularity in the Recommendation Engine for
  universities willing to upload per-course exam records instead of the
  generic aggregate template.
- Wire the Streamlit demo's Excel upload into the production FastAPI
  backend as a real `/api/upload` batch-scoring endpoint.

---

## Original system credits

Built, debugged, integrated, and deployed by **Asadullah Chandio**, Team
Leader, Data Science Alpha Team. The prediction modules started as early
drafts from the Data Science Alpha Team members before being fixed,
rebuilt, and combined into the working system this upgrade builds on top of.

### Original "Running it yourself" (production dashboard)

```bash
npm i -g vercel
git clone <this-repo>
cd <this-repo>
vercel dev
```
Then open `http://localhost:3000`. No environment variables or database
needed — everything the FastAPI backend needs (models, dataset, forecast
data) is already in the repo.
