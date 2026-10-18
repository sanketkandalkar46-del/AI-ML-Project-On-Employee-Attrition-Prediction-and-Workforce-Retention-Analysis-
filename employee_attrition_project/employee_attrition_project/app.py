import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "reports" / "high_risk_employees.csv"
MODEL_PATH = ROOT / "models" / "attrition_model.joblib"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    padding: 18px; border-radius: 12px;
    border: 1px solid rgba(128,128,128,.2);
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

if not CSV_PATH.exists():
    st.error(f"Employee CSV not found: {CSV_PATH}")
    st.stop()

df = load_data()
model_bundle = load_model()

st.sidebar.title("📊 Attrition AI")
st.sidebar.caption("Employee Retention Intelligence")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Employee Explorer", "Risk Analysis", "Model Status"]
)
st.sidebar.divider()
st.sidebar.caption("Data source: high_risk_employees.csv")
st.sidebar.caption(f"Records loaded: {len(df):,}")

required = {"Department", "JobRole", "MonthlyIncome", "Risk_Score"}
missing = required - set(df.columns)
if missing:
    st.error(f"Required columns missing from CSV: {', '.join(missing)}")
    st.stop()

data = df.copy()
data["Risk_Score"] = pd.to_numeric(data["Risk_Score"], errors="coerce")
data["MonthlyIncome"] = pd.to_numeric(data["MonthlyIncome"], errors="coerce")
data = data.dropna(subset=["Risk_Score"])
data["Risk_Level"] = pd.cut(
    data["Risk_Score"],
    bins=[-float("inf"), 0.4, 0.7, float("inf")],
    labels=["Low", "Medium", "High"]
)

if "Actual_Attrition" in data.columns:
    data["Actual_Attrition"] = pd.to_numeric(
        data["Actual_Attrition"], errors="coerce"
    )

def metric_row(frame):
    total = len(frame)
    high = int((frame["Risk_Score"] >= 0.7).sum())
    avg = frame["Risk_Score"].mean() if total else 0
    income = frame["MonthlyIncome"].mean() if total else 0
    cols = st.columns(4)
    cols[0].metric("Employees", f"{total:,}")
    cols[1].metric("High Risk", f"{high:,}", f"{high / total:.1%} of filtered records" if total else "No records")
    cols[2].metric("Average Risk Score", f"{avg:.1%}")
    cols[3].metric("Average Monthly Income", f"{income:,.0f}")

def attrition_rate(frame):
    if "Actual_Attrition" not in frame.columns:
        return None
    valid = frame["Actual_Attrition"].dropna()
    if len(valid) == 0:
        return None
    return (valid == 1).mean()

if page == "Overview":
    st.title("Employee Attrition Intelligence")
    st.write("Understand employee risk, explore workforce patterns, and identify retention priorities.")
    st.info("Dashboard uses the risk scores already stored in your CSV. It does not generate new ML predictions.")

    metric_row(data)
    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Employee Risk Distribution")
        risk_counts = data["Risk_Level"].value_counts().reindex(
            ["Low", "Medium", "High"], fill_value=0
        ).rename_axis("Risk Level").reset_index(name="Employees")
        fig = px.pie(
            risk_counts, names="Risk Level", values="Employees",
            hole=0.58, color="Risk Level",
            color_discrete_map={"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"}
        )
        fig.update_layout(margin=dict(t=20, b=10, l=10, r=10), legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            data, x="Risk_Score", nbins=20,
            labels={"Risk_Score": "Existing Risk Score", "count": "Employees"},
            color_discrete_sequence=["#6366f1"]
        )
        fig.update_layout(margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Employees by Department")
        dept = data.groupby("Department", as_index=False).size().sort_values("size", ascending=False)
        fig = px.bar(dept, x="Department", y="size", color="size",
                     labels={"size": "Employees"}, color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Average Risk by Department")
        dept_risk = data.groupby("Department", as_index=False, observed=False)["Risk_Score"].mean()
        dept_risk = dept_risk.sort_values("Risk_Score", ascending=True)
        fig = px.bar(dept_risk, x="Risk_Score", y="Department", orientation="h",
                     labels={"Risk_Score": "Average Risk Score"},
                     color="Risk_Score", color_continuous_scale="OrRd")
        fig.update_layout(margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-Risk Employees")
    top = data.sort_values("Risk_Score", ascending=False).head(10)
    columns = [c for c in [
        "EmployeeNumber", "Department", "JobRole", "MonthlyIncome",
        "Risk_Score", "Actual_Attrition", "Top_Risk_Drivers"
    ] if c in top.columns]
    st.dataframe(top[columns], use_container_width=True, hide_index=True)

elif page == "Employee Explorer":
    st.title("Employee Explorer")
    st.write("Filter the existing employee records and export the results.")

    c1, c2, c3 = st.columns(3)
    departments = sorted(data["Department"].dropna().astype(str).unique())
    roles = sorted(data["JobRole"].dropna().astype(str).unique())
    selected_depts = c1.multiselect("Department", departments, default=departments)
    selected_roles = c2.multiselect("Job Role", roles, default=roles)
    risk_filter = c3.multiselect("Risk Level", ["Low", "Medium", "High"],
                                 default=["Low", "Medium", "High"])

    filtered = data[
        data["Department"].astype(str).isin(selected_depts)
        & data["JobRole"].astype(str).isin(selected_roles)
        & data["Risk_Level"].astype(str).isin(risk_filter)
    ].copy()

    metric_row(filtered)
    show_cols = [c for c in [
        "EmployeeNumber", "Department", "JobRole", "MonthlyIncome",
        "Risk_Score", "Risk_Level", "Actual_Attrition", "Top_Risk_Drivers"
    ] if c in filtered.columns]

    st.dataframe(
        filtered[show_cols].sort_values("Risk_Score", ascending=False),
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "⬇️ Download filtered employees (CSV)",
        data=filtered[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_employee_attrition.csv",
        mime="text/csv"
    )

elif page == "Risk Analysis":
    st.title("Risk Analysis")
    st.write("Explore how stored risk scores vary across employee groups.")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk by Job Role")
        role_risk = data.groupby("JobRole", as_index=False)["Risk_Score"].mean()
        role_risk = role_risk.sort_values("Risk_Score", ascending=True)
        fig = px.bar(role_risk, x="Risk_Score", y="JobRole", orientation="h",
                     labels={"Risk_Score": "Average Risk Score"},
                     color="Risk_Score", color_continuous_scale="OrRd")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Monthly Income vs Risk")
        fig = px.scatter(
            data, x="MonthlyIncome", y="Risk_Score",
            color="Department", hover_data=[
                c for c in ["EmployeeNumber", "JobRole", "Top_Risk_Drivers"]
                if c in data.columns
            ],
            labels={"MonthlyIncome": "Monthly Income", "Risk_Score": "Existing Risk Score"}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Recorded Risk Drivers")
    if "Top_Risk_Drivers" in data.columns:
        drivers = data["Top_Risk_Drivers"].dropna().astype(str).str.split(", ").explode().str.strip()
        driver_counts = drivers.value_counts().head(12).rename_axis("Risk Driver").reset_index(name="Occurrences")
        fig = px.bar(driver_counts.sort_values("Occurrences"), x="Occurrences", y="Risk Driver",
                     orientation="h", color="Occurrences", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("The CSV does not contain a Top_Risk_Drivers column.")

    rate = attrition_rate(data)
    if rate is not None:
        st.metric("Observed Attrition Rate in This CSV", f"{rate:.1%}")
        st.caption("This is calculated from Actual_Attrition in the supplied file, not from a new prediction.")

elif page == "Model Status":
    st.title("Trained Model Status")
    st.write("Checking the saved model bundle and its contents.")

    if model_bundle is None:
        st.error(f"Model file not found at {MODEL_PATH}")
    else:
        st.success(f"Model bundle loaded: {MODEL_PATH.name}")
        st.write("Loaded object type:", type(model_bundle).__name__)

        if isinstance(model_bundle, dict):
            st.write("Bundle keys:", list(model_bundle.keys()))
            st.warning(
                "The saved file is a dictionary, not directly a prediction model. "
                "No new predictions will be claimed until we identify the actual estimator and its input features."
            )
            for key, value in model_bundle.items():
                st.write(f"**{key}** — {type(value).__name__}")
                if hasattr(value, "predict"):
                    st.success(f"Found a prediction-capable object under key: {key}")
                    st.caption("Input feature requirements still need to be verified before using it.")
        else:
            st.write("Supports predict:", hasattr(model_bundle, "predict"))
            st.write("Supports predict_proba:", hasattr(model_bundle, "predict_proba"))
            st.write("Expected features:", getattr(model_bundle, "feature_names_in_", None))

    st.subheader("Data validation")
    st.write("Rows:", len(data))
    st.write("Columns:", data.columns.tolist())
    st.write("Missing values by column:")
    st.dataframe(data.isna().sum().rename("Missing values"), use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Built with Streamlit • Existing data only")
