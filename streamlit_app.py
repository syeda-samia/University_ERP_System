import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.predict import ModelLoadError, load_model_bundle, run_all_modules
from src.schema import MODULE_LABELS
from src.validation import (
    UnsupportedFileTypeError,
    clean_and_coerce,
    read_uploaded_file,
    validate_upload,
    validate_uploaded_file,
    get_sample_template,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "erp_models.pkl"
SAMPLE_TEMPLATE_PATH = BASE_DIR / "sample_data" / "sample_students_template.xlsx"

st.set_page_config(page_title="University ERP Predictive Analytics", page_icon="🎓", layout="wide")

CUSTOM_CSS = """
<style>
    :root { --accent: #FF1E27; }
    .stApp { background-color: #0E0E10; color: #F2F2F2; }
    h1, h2, h3, h4 { color: #FFFFFF !important; }
    .app-title { color: var(--accent) !important; }
    .stButton > button, .stDownloadButton > button {
        background-color: var(--accent);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        transition: background-color 0.2s ease-in-out;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #cc1820;
        color: #FFFFFF;
    }
    .module-card {
        background-color: #1A1A1D;
        border: 1px solid #2A2A2E;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .module-card b { color: var(--accent); }
    .footer {
        text-align: center;
        color: #888888;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2A2A2E;
        font-size: 0.9rem;
    }
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #1A1A1D;
        border-color: #2A2A2E;
    }
    .validation-success {
        background-color: #1A3A1A;
        border: 1px solid #2ECC71;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .validation-error {
        background-color: #3A1A1A;
        border: 1px solid #FF1E27;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .validation-warning {
        background-color: #3A2A1A;
        border: 1px solid #F5A623;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#FF1E27"
RISK_COLORS = {"Low": "#2ECC71", "Medium": "#F5A623", "High": "#FF1E27"}


@st.cache_resource
def get_model_bundle():
    return load_model_bundle(MODEL_PATH)


try:
    bundle = get_model_bundle()
    model_load_error = None
except ModelLoadError as exc:
    bundle = None
    model_load_error = str(exc)

st.markdown('<h1 class="app-title">🎓 University ERP — Predictive Analytics Demo</h1>', unsafe_allow_html=True)

if model_load_error:
    st.error(f"The prediction models could not be loaded: {model_load_error}")
    st.stop()

st.markdown(
    """
This is a portfolio demo of a predictive analytics layer for a university ERP system,
covering **6 modules** in one place. Upload your own student data (any university's
own Excel/CSV export) or try the sample dataset below — the same 6 models run on
whichever columns are present.
"""
)

with st.container():
    cols = st.columns(3)
    descriptions = [
        ("🚨 Student Risk", "Flags students likely to fail this term, from attendance, CGPA, assignment completion, and backlogs."),
        ("📉 Dropout Prediction", "Estimates dropout risk from attendance, CGPA, fee payment status, and LMS activity."),
        ("💸 Fee Default", "Predicts the chance a student defaults on tuition, from demographics, department, and CGPA."),
        ("📈 GPA Prediction", "Forecasts next-term GPA from attendance, quiz/assignment/exam performance, and current CGPA."),
        ("🎯 Recommendations", "Flags which academic area (quizzes, assignments, exams, labs) a student is weakest in."),
        ("🏛️ Enrollment Forecast", "Projects future enrollment from historical year-over-year enrollment counts."),
    ]
    for i, (title, desc) in enumerate(descriptions):
        with cols[i % 3]:
            st.markdown(f'<div class="module-card"><b>{title}</b><br/><span style="color:#AAAAAA; font-size: 0.9rem;">{desc}</span></div>', unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Data input: upload or sample dataset
# ---------------------------------------------------------------------------
st.subheader("1. Provide student data")

col_mode, col_empty = st.columns([2, 1])
with col_mode:
    strict_mode = st.checkbox(
        "🔒 Strict ERP Format",
        value=False,
        help="Enable: Only accept exact University ERP export format. Disable: Auto-detect and map columns."
    )
    if strict_mode:
        st.caption("⚠️ **Strict mode:** All required columns must match exactly.")
    else:
        st.caption("✅ **Flexible mode:** Columns will be auto-detected and mapped.")

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload your own student data (.xlsx or .csv)", type=["xlsx", "xls", "csv"]
    )

with col_sample:
    st.write("No data handy?")
    use_sample = st.button("▶ Try the sample dataset", width="stretch")

    if SAMPLE_TEMPLATE_PATH.exists():
        with open(SAMPLE_TEMPLATE_PATH, "rb") as f:
            st.download_button(
                "⬇ Download sample template (Excel)",
                data=f.read(),
                file_name="sample_students_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )
    else:
        sample_df = get_sample_template()
        csv_data = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇ Download sample template (CSV)",
            data=csv_data,
            file_name="sample_students_template.csv",
            mime="text/csv",
            width="stretch",
        )

raw_df = None
source_label = None
validation_message = None
is_valid = False
report = None

if use_sample:
    if SAMPLE_TEMPLATE_PATH.exists():
        with open(SAMPLE_TEMPLATE_PATH, "rb") as f:
            is_valid, raw_df, validation_message = validate_uploaded_file(f, strict_mode=strict_mode)
            source_label = "sample dataset"
    else:
        st.error("Sample dataset not found.")
elif uploaded_file is not None:
    is_valid, raw_df, validation_message = validate_uploaded_file(uploaded_file, strict_mode=strict_mode)
    source_label = uploaded_file.name

if validation_message:
    st.markdown("---")
    if "❌" in validation_message or "Validation Failed" in validation_message:
        st.markdown(f'<div class="validation-error">{validation_message}</div>', unsafe_allow_html=True)
    elif "⚠️" in validation_message or "warning" in validation_message.lower():
        st.markdown(f'<div class="validation-warning">{validation_message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="validation-success">{validation_message}</div>', unsafe_allow_html=True)
    st.markdown("---")

# ============================================
# Process valid data - WITH PROPER REPORT HANDLING
# ============================================
if raw_df is not None and is_valid:
    # Show data preview
    with st.expander("📊 Data Preview", expanded=False):
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.caption(f"Total rows: {len(raw_df)} | Columns: {len(raw_df.columns)}")

    # ============================================
    # CRITICAL FIX: Get validation report
    # ============================================
    from src.validation import validate_upload
    report = validate_upload(raw_df)

    # ============================================
    # CHECK: Enrollment Module Skip Reason
    # ============================================
    enrollment_skip_reason = None

    # Check if EnrollmentDate exists
    if "EnrollmentDate" in raw_df.columns:
        # Check unique years
        try:
            dates = pd.to_datetime(raw_df["EnrollmentDate"]).dropna()
            unique_years = dates.dt.year.nunique()
            
            if unique_years < 2:
                enrollment_skip_reason = f"⚠️ **Enrollment Forecasting is not available:** Only {unique_years} year(s) of enrollment data found. Need at least 2 different years for trend analysis."
            else:
                enrollment_skip_reason = f"✅ **Enrollment Forecasting:** {unique_years} years of data available."
        except Exception:
            enrollment_skip_reason = "⚠️ **Enrollment Forecasting is not available:** Unable to parse EnrollmentDate column."
    else:
        enrollment_skip_reason = "⚠️ **Enrollment Forecasting is not available:** 'EnrollmentDate' column is missing."

    # Show the reason to user
    if enrollment_skip_reason:
        if "⚠️" in enrollment_skip_reason:
            st.warning(enrollment_skip_reason)
        elif "✅" in enrollment_skip_reason:
            st.info(enrollment_skip_reason)
    
    # Debug: Show what modules are available
    st.write(f"🔍 **Debug:** Available modules from report: {report.available_modules}")

    if not report.available_modules:
        st.warning("⚠️ No modules available for this data.")
        st.markdown("**Required columns for each module:**")
        from src.schema import MODULE_REQUIRED_COLUMNS
        for module, cols in MODULE_REQUIRED_COLUMNS.items():
            st.markdown(f"- **{MODULE_LABELS.get(module, module)}**: `{', '.join(cols)}`")
    else:
        with st.spinner(f"Running predictions across {len(report.available_modules)} modules..."):
            try:
                cleaned = clean_and_coerce(raw_df)

                # Run predictions
                results = run_all_modules(cleaned, bundle, report.available_modules)

            except Exception as e:
                logger.exception("Prediction pipeline failed")
                st.error(f"❌ **Prediction Error:** {str(e)}")
                import traceback
                st.code(traceback.format_exc(), language="python")
                results = {}

            if results:
                st.divider()
                st.subheader("2. Results")

                tab_labels = [MODULE_LABELS.get(m, m) for m in results.keys()]
                tabs = st.tabs(tab_labels)

                for tab, module_name in zip(tabs, results.keys()):
                    with tab:
                        result = results[module_name]

                        if module_name == "student_risk":
                            st.dataframe(result, hide_index=True, use_container_width=True)
                            fig = px.pie(result, names="risk_level", title="Risk level distribution",
                                         color="risk_level", color_discrete_map=RISK_COLORS, template=PLOTLY_TEMPLATE)
                            st.plotly_chart(fig, use_container_width=True)

                        elif module_name == "dropout":
                            st.dataframe(result, hide_index=True, use_container_width=True)
                            fig = px.pie(result, names="risk_level", title="Dropout risk distribution",
                                         color="risk_level", color_discrete_map=RISK_COLORS, template=PLOTLY_TEMPLATE)
                            st.plotly_chart(fig, use_container_width=True)

                        elif module_name == "fee_default":
                            st.dataframe(result, hide_index=True, use_container_width=True)
                            fig = px.pie(result, names="risk_level", title="Fee default risk distribution",
                                         color="risk_level", color_discrete_map=RISK_COLORS, template=PLOTLY_TEMPLATE)
                            st.plotly_chart(fig, use_container_width=True)

                        elif module_name == "gpa":
                            st.dataframe(result, hide_index=True, use_container_width=True)
                            fig = px.histogram(result, x="predicted_gpa", nbins=20, title="Predicted GPA distribution",
                                                color_discrete_sequence=[ACCENT], template=PLOTLY_TEMPLATE)
                            st.plotly_chart(fig, use_container_width=True)

                        elif module_name == "recommendation":
                            st.dataframe(result, hide_index=True, use_container_width=True)
                            if "status" in result.columns:
                                fig = px.bar(result["status"].value_counts().reset_index(),
                                            x="status", y="count",
                                            title="Students needing attention",
                                            color_discrete_sequence=[ACCENT],
                                            template=PLOTLY_TEMPLATE)
                                st.plotly_chart(fig, use_container_width=True)

                        elif module_name == "enrollment_forecast":
                            hist = result["historical"]
                            fcast = result["forecast"]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=list(hist.keys()), y=list(hist.values()),
                                                      mode="lines+markers", name="Historical",
                                                      line=dict(color="#5AC8FA")))
                            fig.add_trace(go.Scatter(x=list(fcast.keys()), y=list(fcast.values()),
                                                      mode="lines+markers", name="Forecast",
                                                      line=dict(color=ACCENT, dash="dash")))
                            fig.update_layout(title=f"Enrollment: historical + forecast (trend R²={result['trend_r2']:.2f})",
                                               template=PLOTLY_TEMPLATE,
                                               xaxis_title="Year",
                                               yaxis_title="Students enrolled")
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(
                                "Forecast is a simple linear trend on real historical enrollment counts "
                                "(lightweight by design — see README for why Prophet wasn't used)."
                            )
            else:
                st.warning("⚠️ No predictions could be generated for any module.")
                st.markdown("**Required columns for each module:**")
                from src.schema import MODULE_REQUIRED_COLUMNS
                for module, cols in MODULE_REQUIRED_COLUMNS.items():
                    present = [c for c in cols if c in raw_df.columns]
                    missing = [c for c in cols if c not in raw_df.columns]
                    status = "✅" if len(present) == len(cols) else "❌"
                    st.markdown(f"- **{MODULE_LABELS.get(module, module)}** {status}")
                    if missing:
                        st.markdown(f"  *Missing: {', '.join(missing)}*")

elif raw_df is None and not use_sample and uploaded_file is None:
    st.info("👆 Upload a file or click 'Try the sample dataset' to get started.")

st.divider()

# ---------------------------------------------------------------------------
# How it works
# ---------------------------------------------------------------------------
with st.expander("📊 How it works — model performance & methodology"):
    st.markdown(f"Models trained on **{bundle.get('n_training_students', '?')} students** from real attendance and exam records.")

    rows = []
    for module in ["student_risk", "dropout", "fee_default"]:
        m = bundle[module]["test_metrics"]
        rows.append({
            "Module": MODULE_LABELS[module],
            "Model": bundle[module]["model_name"],
            "Accuracy": f"{m['accuracy']:.2f}",
            "Precision": f"{m['precision']:.2f}",
            "Recall": f"{m['recall']:.2f}",
            "F1": f"{m['f1']:.2f}",
        })
    st.write("**Classification modules (held-out test set):**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    gpa_metrics = bundle["gpa"]["test_metrics"]
    st.write("**GPA regression module (held-out test set):**")
    st.dataframe(pd.DataFrame([{
        "Model": bundle["gpa"]["model_name"],
        "R²": f"{gpa_metrics['r2']:.3f}",
        "RMSE": f"{gpa_metrics['rmse']:.3f}",
        "MAE": f"{gpa_metrics['mae']:.3f}",
    }]), hide_index=True, use_container_width=True)

    st.markdown(
        """
**Honest notes on methodology:**
- Student Risk and GPA targets are derived from real attendance/exam data.
- Dropout and Fee Default labels are documented, simulated rules (the source
  dataset has no real dropout outcomes or fee-transaction records) — see
  README "Model methodology" for exactly how and why.
- All 4 ML modules use 5-fold cross-validation for model selection and a
  held-out test set (never seen during training or tuning) for the metrics above.
"""
    )

st.markdown('<div class="footer">Built by Vexanex Digital Solutions</div>', unsafe_allow_html=True)
