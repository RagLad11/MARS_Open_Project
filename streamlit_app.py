import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from predict import (
    apply_pseudo_labeler,
    build_classifier_features,
    load_and_prepare_for_prediction,
    make_dossier,
)
from train_pipeline import SEVERITY_INV


st.set_page_config(page_title="Support Integrity Auditor", layout="wide")

MODEL_DIR = Path("models")


@st.cache_resource
def load_artifacts(model_dir):
    pseudo_path = model_dir / "pseudo_label_artifacts.joblib"
    clf_path = model_dir / "classifier_artifacts.joblib"
    if not pseudo_path.exists() or not clf_path.exists():
        return None, None
    return joblib.load(pseudo_path), joblib.load(clf_path)


def run_sia(df_input, pseudo, clf_artifacts):
    with NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        df_input.to_csv(tmp.name, index=False)
        prepared = load_and_prepare_for_prediction(tmp.name)

    prepared = apply_pseudo_labeler(prepared, pseudo)
    x = build_classifier_features(prepared, clf_artifacts)
    probs = clf_artifacts["classifier"].predict_proba(x)[:, 1]
    preds = (probs >= clf_artifacts["threshold"]).astype(int)

    result = prepared.copy()
    result["predicted_mismatch"] = preds
    result["mismatch_probability"] = probs
    result["assigned_priority"] = result["assigned_ordinal"].map(SEVERITY_INV)
    result["inferred_severity"] = result["inferred_severity_ordinal"].map(SEVERITY_INV)
    result["mismatch_type"] = result["severity_delta"].apply(
        lambda d: "Hidden Crisis" if d > 0 else ("False Alarm" if d < 0 else "Consistent")
    )

    dossiers = []
    for i, row in result.iterrows():
        if int(row["predicted_mismatch"]) == 1 and int(row["severity_delta"]) != 0:
            dossiers.append(make_dossier(row, probs[i], pseudo["keywords"]))
    return result, dossiers


def single_ticket_frame():
    with st.form("single_ticket_form"):
        col1, col2 = st.columns(2)
        with col1:
            ticket_id = st.text_input("Ticket ID", "TKT-DEMO-001")
            subject = st.text_input("Ticket Subject", "Dashboard not loading data")
            priority = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"], index=1)
            channel = st.selectbox("Ticket Channel", ["Email", "Chat", "Web Form"], index=1)
        with col2:
            category = st.selectbox("Issue Category", ["Technical", "Billing", "Account", "General Inquiry", "Fraud"])
            email = st.text_input("Customer Email", "customer@example.com")
            resolution_time = st.number_input("Resolution Time Hours", min_value=0.0, value=48.0, step=1.0)
        description = st.text_area(
            "Ticket Description",
            "Hi Support, The dashboard is not loading any data, just a spinning wheel.",
            height=120,
        )
        submitted = st.form_submit_button("Audit Ticket")

    if not submitted:
        return None

    return pd.DataFrame(
        [
            {
                "Ticket_ID": ticket_id,
                "Customer_Email": email,
                "Ticket_Subject": subject,
                "Ticket_Description": description,
                "Issue_Category": category,
                "Priority_Level": priority,
                "Ticket_Channel": channel,
                "Resolution_Time_Hours": resolution_time,
            }
        ]
    )


def show_dossiers(dossiers):
    if not dossiers:
        st.info("No mismatch dossiers were generated for the current input.")
        return
    for dossier in dossiers:
        with st.expander(f"{dossier['ticket_id']} - {dossier['mismatch_type']}"):
            st.json(dossier)


def show_dashboard(result):
    dash1, dash2, dash3 = st.columns(3)
    flagged = int(result["predicted_mismatch"].sum())
    dash1.metric("Tickets Audited", len(result))
    dash2.metric("Flagged Mismatches", flagged)
    dash3.metric("Flag Rate", f"{flagged / max(len(result), 1):.1%}")

    col1, col2 = st.columns(2)
    with col1:
        counts = result["predicted_mismatch"].map({0: "Consistent", 1: "Mismatch"}).value_counts().reset_index()
        counts.columns = ["Judgment", "Count"]
        st.plotly_chart(px.bar(counts, x="Judgment", y="Count", title="Priority Mismatch Distribution"), use_container_width=True)

    with col2:
        mismatch_counts = result[result["predicted_mismatch"] == 1]["mismatch_type"].value_counts().reset_index()
        mismatch_counts.columns = ["Mismatch Type", "Count"]
        st.plotly_chart(px.bar(mismatch_counts, x="Mismatch Type", y="Count", title="Mismatch Types"), use_container_width=True)

    st.subheader("Top Contributing Signals")
    signal_cols = ["sig_a_severity", "sig_b_rt_proxy", "sig_c_rules", "sig_d_cluster"]
    signal_df = result[signal_cols].mean().reset_index()
    signal_df.columns = ["Signal", "Average Severity"]
    signal_df["Signal"] = signal_df["Signal"].replace(
        {
            "sig_a_severity": "SBERT semantic anchors",
            "sig_b_rt_proxy": "Resolution-time proxy",
            "sig_c_rules": "Rule-based NLP",
            "sig_d_cluster": "Embedding clusters",
        }
    )
    st.plotly_chart(px.bar(signal_df, x="Average Severity", y="Signal", orientation="h"), use_container_width=True)

    st.subheader("Severity Delta Heatmap")
    heat = result.pivot_table(
        values="severity_delta",
        index="Issue_Category",
        columns="Ticket_Channel",
        aggfunc="mean",
        fill_value=0,
    )
    st.plotly_chart(
        px.imshow(
            heat,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Average Severity Delta by Category and Channel",
            aspect="auto",
        ),
        use_container_width=True,
    )


def main():
    st.title("Support Integrity Auditor")
    st.caption("Semantics-driven priority mismatch detection with grounded evidence dossiers.")

    pseudo, clf_artifacts = load_artifacts(MODEL_DIR)
    if pseudo is None or clf_artifacts is None:
        st.error(
            "Model artifacts were not found. Run `python train_pipeline.py --data customer_support_tickets.csv "
            "--model-dir models --outputs-dir outputs` before starting the app."
        )
        st.stop()

    tab_single, tab_batch = st.tabs(["Single Ticket", "Batch CSV"])

    with tab_single:
        df_single = single_ticket_frame()
        if df_single is not None:
            with st.spinner("Auditing ticket..."):
                result, dossiers = run_sia(df_single, pseudo, clf_artifacts)
            row = result.iloc[0]
            judgment = "Mismatch" if row["predicted_mismatch"] == 1 else "Consistent"
            st.metric("Binary Judgment", judgment, f"confidence {row['mismatch_probability']:.2%}")
            st.dataframe(
                result[
                    [
                        "Ticket_ID",
                        "Priority_Level",
                        "inferred_severity",
                        "severity_delta",
                        "mismatch_type",
                        "mismatch_probability",
                    ]
                ],
                use_container_width=True,
            )
            show_dossiers(dossiers)

    with tab_batch:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            df_batch = pd.read_csv(uploaded)
            with st.spinner("Auditing CSV..."):
                result, dossiers = run_sia(df_batch, pseudo, clf_artifacts)

            show_dashboard(result)
            st.subheader("Predictions")
            view_cols = [
                "Ticket_ID",
                "Priority_Level",
                "Ticket_Channel",
                "Issue_Category",
                "Resolution_Time_Hours",
                "inferred_severity",
                "severity_delta",
                "mismatch_type",
                "predicted_mismatch",
                "mismatch_probability",
            ]
            st.dataframe(result[view_cols], use_container_width=True)

            csv_bytes = result[view_cols].to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions CSV", csv_bytes, "sia_predictions.csv", "text/csv")
            st.download_button(
                "Download Dossiers JSON",
                json.dumps(dossiers, indent=2).encode("utf-8"),
                "sia_dossiers.json",
                "application/json",
            )
            show_dossiers(dossiers)


if __name__ == "__main__":
    main()
