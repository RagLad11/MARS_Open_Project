import csv
import io
import json
from collections import Counter, defaultdict

import streamlit as st


st.set_page_config(page_title="Support Integrity Auditor", layout="wide")

SEVERITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
SEVERITY_INV = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


def clean_key(name):
    return str(name).strip().replace(" ", "_")


def read_csv_file(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for i, row in enumerate(reader):
        clean = {clean_key(k): v for k, v in row.items()}
        if not clean.get("Ticket_ID"):
            clean["Ticket_ID"] = f"TKT-{i:06d}"
        rows.append(clean)
    return rows


def require_columns(rows):
    required = [
        "Ticket_Subject",
        "Ticket_Description",
        "Issue_Category",
        "Priority_Level",
        "Ticket_Channel",
        "Resolution_Time_Hours",
    ]
    if not rows:
        raise ValueError("CSV is empty.")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def infer_severity(row):
    text = f"{row.get('Ticket_Subject', '')} {row.get('Ticket_Description', '')} {row.get('Issue_Category', '')}".lower()
    try:
        rt = float(row.get("Resolution_Time_Hours", 0))
    except ValueError:
        rt = 0.0

    score = 0
    evidence = []

    critical_words = ["outage", "security breach", "breach", "data loss", "production down", "fraud"]
    high_words = ["cannot login", "crash", "crashes", "not loading", "payment", "failed", "urgent", "escalate"]
    low_words = ["question", "how do i", "hours of operation", "where is", "feature request", "clarification"]
    neg_words = ["not urgent", "resolved", "false alarm", "workaround"]

    for label, words, points, meaning in [
        ("keyword", critical_words, 3, "critical phrase found in ticket text"),
        ("keyword", high_words, 2, "high urgency phrase found in ticket text"),
        ("keyword", low_words, -1, "low urgency phrase found in ticket text"),
        ("keyword", neg_words, -2, "negation or false-alarm phrase found in ticket text"),
    ]:
        found = [w for w in words if w in text]
        if found:
            score += points
            evidence.append({"signal": label, "value": ", ".join(found), "interpretation": meaning})

    if rt >= 96:
        score += 2
        evidence.append({"signal": "resolution_time", "value": f"{rt:.1f} hours", "interpretation": "very long resolution time"})
    elif rt >= 48:
        score += 1
        evidence.append({"signal": "resolution_time", "value": f"{rt:.1f} hours", "interpretation": "moderate to long resolution time"})
    elif rt <= 12:
        score -= 1
        evidence.append({"signal": "resolution_time", "value": f"{rt:.1f} hours", "interpretation": "short resolution time"})

    category = str(row.get("Issue_Category", "")).strip().lower()
    if category == "fraud":
        score += 2
        evidence.append({"signal": "issue_category", "value": "Fraud", "interpretation": "fraud tickets are treated as higher risk"})
    elif category == "technical":
        score += 1
        evidence.append({"signal": "issue_category", "value": "Technical", "interpretation": "technical tickets may block product usage"})

    if score >= 4:
        return 3, evidence
    if score >= 2:
        return 2, evidence
    if score >= 0:
        return 1, evidence
    return 0, evidence


def audit_rows(rows):
    require_columns(rows)
    results = []
    dossiers = []

    for i, row in enumerate(rows):
        priority = str(row.get("Priority_Level", "Medium")).strip().title()
        assigned = SEVERITY_MAP.get(priority, 1)
        inferred, evidence = infer_severity(row)
        delta = inferred - assigned
        mismatch = abs(delta) >= 1
        mismatch_type = "Consistent"
        if delta > 0:
            mismatch_type = "Hidden Crisis"
        elif delta < 0:
            mismatch_type = "False Alarm"

        confidence = min(0.95, 0.55 + 0.15 * abs(delta) + 0.03 * len(evidence))
        ticket_id = row.get("Ticket_ID") or f"TKT-{i:06d}"
        result = {
            "Ticket_ID": ticket_id,
            "Priority_Level": priority,
            "Ticket_Channel": row.get("Ticket_Channel", ""),
            "Issue_Category": row.get("Issue_Category", ""),
            "Resolution_Time_Hours": row.get("Resolution_Time_Hours", ""),
            "assigned_priority": SEVERITY_INV[assigned],
            "inferred_severity": SEVERITY_INV[inferred],
            "severity_delta": delta,
            "predicted_mismatch": "Mismatch" if mismatch else "Consistent",
            "mismatch_type": mismatch_type,
            "confidence": round(confidence, 3),
        }
        results.append(result)

        if mismatch:
            if not evidence:
                evidence = [
                    {
                        "signal": "priority_comparison",
                        "value": f"{SEVERITY_INV[assigned]} vs {SEVERITY_INV[inferred]}",
                        "interpretation": "assigned priority differs from inferred severity",
                    }
                ]
            dossiers.append(
                {
                    "ticket_id": ticket_id,
                    "assigned_priority": SEVERITY_INV[assigned],
                    "inferred_severity": SEVERITY_INV[inferred],
                    "mismatch_type": mismatch_type,
                    "severity_delta": f"+{delta}" if delta > 0 else str(delta),
                    "feature_evidence": evidence,
                    "constraint_analysis": (
                        f"The ticket subject is '{row.get('Ticket_Subject', '')}'. "
                        f"The assigned priority is {SEVERITY_INV[assigned]}, while the auditor inferred "
                        f"{SEVERITY_INV[inferred]} from ticket text, issue category, channel, and resolution time."
                    ),
                    "confidence": round(confidence, 3),
                }
            )

    return results, dossiers


def to_csv(rows):
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def show_dashboard(results):
    st.subheader("Priority Mismatch Dashboard")
    total = len(results)
    mismatches = sum(1 for r in results if r["predicted_mismatch"] == "Mismatch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tickets", total)
    c2.metric("Flagged Mismatches", mismatches)
    c3.metric("Flag Rate", f"{mismatches / max(total, 1):.1%}")

    left, right = st.columns(2)
    with left:
        st.write("Distribution of Flagged Tickets")
        st.bar_chart(Counter(r["predicted_mismatch"] for r in results))
    with right:
        st.write("Mismatch Types")
        st.bar_chart(Counter(r["mismatch_type"] for r in results))

    st.write("Top Contributing Signals")
    signal_counts = Counter()
    for r in results:
        if r["Resolution_Time_Hours"]:
            try:
                rt = float(r["Resolution_Time_Hours"])
                if rt >= 48 or rt <= 12:
                    signal_counts["resolution_time"] += 1
            except ValueError:
                pass
        if r["Issue_Category"] in ["Fraud", "Technical"]:
            signal_counts["issue_category"] += 1
        if r["severity_delta"] != 0:
            signal_counts["priority_comparison"] += 1
    st.bar_chart(signal_counts)

    st.write("Severity Delta Heatmap")
    matrix = defaultdict(dict)
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["Issue_Category"], r["Ticket_Channel"])].append(r["severity_delta"])
    for (category, channel), values in grouped.items():
        matrix[category][channel] = round(sum(values) / len(values), 2)
    st.dataframe(dict(matrix), use_container_width=True)


st.title("Support Integrity Auditor")
st.caption("Simple priority mismatch auditor with evidence dossiers.")

tab1, tab2 = st.tabs(["Single Ticket", "Batch CSV"])

with tab1:
    with st.form("ticket_form"):
        subject = st.text_input("Ticket Subject", "Dashboard not loading data")
        description = st.text_area("Ticket Description", "The dashboard is not loading any data, just a spinning wheel.")
        priority = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"], index=1)
        channel = st.selectbox("Ticket Channel", ["Email", "Chat", "Web Form"], index=1)
        category = st.selectbox("Issue Category", ["Technical", "Billing", "Account", "General Inquiry", "Fraud"])
        resolution_time = st.number_input("Resolution Time Hours", min_value=0.0, value=48.0)
        submitted = st.form_submit_button("Audit Ticket")

    if submitted:
        rows = [
            {
                "Ticket_ID": "FORM-001",
                "Ticket_Subject": subject,
                "Ticket_Description": description,
                "Issue_Category": category,
                "Priority_Level": priority,
                "Ticket_Channel": channel,
                "Resolution_Time_Hours": str(resolution_time),
            }
        ]
        results, dossiers = audit_rows(rows)
        st.metric("Binary Judgment", results[0]["predicted_mismatch"], f"confidence {results[0]['confidence']:.0%}")
        st.dataframe(results, use_container_width=True)
        st.subheader("Evidence Dossier")
        st.json(dossiers[0] if dossiers else {"message": "No mismatch detected."})

with tab2:
    uploaded = st.file_uploader("Upload customer_support_tickets.csv", type="csv")
    if uploaded:
        try:
            rows = read_csv_file(uploaded)
            results, dossiers = audit_rows(rows)
            show_dashboard(results)
            st.subheader("Predictions")
            st.dataframe(results, use_container_width=True)
            st.download_button("Download Predictions CSV", to_csv(results), "predictions.csv")
            st.download_button("Download Dossiers JSON", json.dumps(dossiers, indent=2), "dossiers.json")
            st.subheader("Evidence Dossiers")
            for dossier in dossiers[:50]:
                with st.expander(f"{dossier['ticket_id']} - {dossier['mismatch_type']}"):
                    st.json(dossier)
            if len(dossiers) > 50:
                st.info(f"Showing first 50 dossiers out of {len(dossiers)}.")
        except Exception as exc:
            st.error(str(exc))
