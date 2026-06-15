import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from train_pipeline import SEVERITY_INV, SEVERITY_MAP, categorize_rt, canonicalize_col, clean_description


def load_and_prepare_for_prediction(path):
    df = pd.read_csv(path)
    df = df.rename(columns={c: canonicalize_col(c) for c in df.columns})

    required = [
        "Ticket_Subject",
        "Ticket_Description",
        "Customer_Email",
        "Priority_Level",
        "Ticket_Channel",
        "Resolution_Time_Hours",
        "Issue_Category",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    if "Product_Purchased" not in df.columns:
        df["Product_Purchased"] = "Unknown Product"
    if "Ticket_ID" not in df.columns:
        df["Ticket_ID"] = [f"TKT-{i:06d}" for i in range(len(df))]

    df["Resolution_Time_Hours"] = pd.to_numeric(df["Resolution_Time_Hours"], errors="coerce")
    df["Resolution_Time_Hours"] = df["Resolution_Time_Hours"].fillna(df["Resolution_Time_Hours"].median())

    priority_aliases = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}
    raw_priority = df["Priority_Level"].astype(str).str.strip()
    df["Priority_Level"] = raw_priority.str.lower().map(priority_aliases).fillna(raw_priority)
    unknown = sorted(set(df["Priority_Level"]) - set(SEVERITY_MAP))
    if unknown:
        raise ValueError(f"Unexpected priority labels: {unknown}")

    df["clean_subject"] = df["Ticket_Subject"].astype(str).str.strip()
    df["clean_desc"] = df["Ticket_Description"].astype(str).apply(clean_description)
    df["combined_text"] = df["clean_subject"] + " [SEP] " + df["clean_desc"]
    df["Customer_Domain"] = (
        df["Customer_Email"].astype(str).str.extract(r"@([^>\s]+)", expand=False).fillna("unknown").str.lower()
    )
    df["Product_Name"] = df["Product_Purchased"].astype(str).fillna("Unknown Product")
    df["RT_Bucket"] = df["Resolution_Time_Hours"].astype(float).apply(categorize_rt)
    df["assigned_ordinal"] = df["Priority_Level"].map(SEVERITY_MAP).astype(int)
    return df


def ordinal_apply(values, q):
    return np.digitize(values, q)


def apply_pseudo_labeler(df, artifacts):
    sbert = SentenceTransformer(artifacts["sbert_name"])
    embeddings = sbert.encode(
        df["combined_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    a_cont = []
    for emb in embeddings:
        sims = {
            lv: float(np.max(cosine_similarity(emb.reshape(1, -1), artifacts["anchor_embeddings"][lv])[0]))
            for lv in artifacts["anchor_embeddings"]
        }
        exp_s = {lv: np.exp(8 * sims[lv]) for lv in sims}
        a_cont.append(sum(lv * exp_s[lv] for lv in exp_s) / sum(exp_s.values()))
    sig_a = ordinal_apply(np.array(a_cont), artifacts["qa"])

    text = artifacts["rt_vectorizer"].transform(df["combined_text"])
    cats = pd.get_dummies(df[["Ticket_Channel", "Issue_Category", "Product_Name"]].astype(str))
    cats = cats.reindex(columns=artifacts["rt_cat_cols"], fill_value=0)
    x_rt = sparse.hstack([text, sparse.csr_matrix(cats.values)], format="csr")
    rt_pred = artifacts["rt_model"].predict(x_rt)
    sig_b = ordinal_apply(rt_pred, artifacts["qb"])

    kws = artifacts["keywords"]
    sig_c = []
    for text_value in df["combined_text"].str.lower():
        score = 1
        if any(k in text_value for k in kws["low"]):
            score -= 1
        if any(k in text_value for k in kws["high"]):
            score += 1
        if any(k in text_value for k in kws["escalation"]):
            score += 1
        if any(k in text_value for k in kws["critical"]):
            score += 2
        if any(k in text_value for k in kws["negation"]):
            score -= 2
        sig_c.append(int(np.clip(score, 0, 3)))
    sig_c = np.array(sig_c)

    clusters = artifacts["kmeans"].predict(embeddings)
    d_cont = np.array([artifacts["cluster_score"][c] for c in clusters])
    sig_d = ordinal_apply(d_cont, artifacts["qd"])

    weights = artifacts["best_weights"]
    fused = sum(w * s for w, s in zip(weights, [sig_a, sig_b, sig_c, sig_d]))
    inferred = ordinal_apply(fused, artifacts["qf"])

    df["inferred_severity_ordinal"] = inferred
    df["fused_severity_score"] = fused
    df["severity_delta"] = df["inferred_severity_ordinal"] - df["assigned_ordinal"]
    df["sig_a_severity"] = sig_a
    df["sig_b_rt_proxy"] = sig_b
    df["sig_c_rules"] = sig_c
    df["sig_d_cluster"] = sig_d
    df["signal_agreement"] = 1.0 - (np.std(np.column_stack([sig_a, sig_b, sig_c, sig_d]), axis=1) / 1.5).clip(0, 1)
    return df


def build_classifier_features(df, classifier_artifacts):
    text = (
        df["combined_text"].fillna("")
        + " assigned_priority="
        + df["Priority_Level"].astype(str)
        + " channel="
        + df["Ticket_Channel"].astype(str)
        + " category="
        + df["Issue_Category"].astype(str)
        + " domain="
        + df["Customer_Domain"].astype(str)
    )
    text_x = classifier_artifacts["vectorizer"].transform(text)
    cats = pd.get_dummies(df[["Priority_Level", "Ticket_Channel", "Issue_Category", "RT_Bucket"]].astype(str))
    cats = cats.reindex(columns=classifier_artifacts["cat_cols"], fill_value=0)
    nums = df[["Resolution_Time_Hours", "assigned_ordinal"]].astype(float).values
    return sparse.hstack([text_x, sparse.csr_matrix(cats.values), sparse.csr_matrix(nums)], format="csr")


def make_dossier(row, confidence, keywords):
    ass_ord = int(row["assigned_ordinal"])
    inf_ord = int(row["inferred_severity_ordinal"])
    delta = inf_ord - ass_ord
    mismatch_type = "Hidden Crisis" if delta > 0 else "False Alarm"
    delta_str = f"+{delta}" if delta > 0 else str(delta)

    text_l = str(row["combined_text"]).lower()
    all_keywords = keywords["critical"] + keywords["high"] + keywords["escalation"] + keywords["low"] + keywords["negation"]
    found = sorted({k for k in all_keywords if k in text_l})

    evidence = [
        {"signal": "assigned_priority", "value": str(row["Priority_Level"]), "weight": "comparison baseline"},
        {
            "signal": "self_supervised_severity",
            "value": SEVERITY_INV[inf_ord],
            "weight": f"fusion={float(row['fused_severity_score']):.2f}",
        },
        {
            "signal": "resolution_time",
            "value": f"{float(row['Resolution_Time_Hours']):.1f} hours",
            "interpretation": f"Input Resolution_Time_Hours falls in {row['RT_Bucket']}.",
        },
        {"signal": "channel", "value": str(row["Ticket_Channel"]), "weight": "metadata feature"},
    ]
    if found:
        evidence.append({"signal": "keyword", "value": ", ".join(found[:6]), "weight": "rule feature"})

    return {
        "ticket_id": str(row["Ticket_ID"]),
        "assigned_priority": SEVERITY_INV[ass_ord],
        "inferred_severity": SEVERITY_INV[inf_ord],
        "mismatch_type": mismatch_type,
        "severity_delta": delta_str,
        "feature_evidence": evidence,
        "constraint_analysis": (
            f"The ticket subject is '{row['Ticket_Subject']}' and the assigned priority is {SEVERITY_INV[ass_ord]}. "
            f"The self-supervised severity estimate is {SEVERITY_INV[inf_ord]}, giving a calibrated severity delta of {delta_str}."
        ),
        "confidence": round(float(confidence), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Support Integrity Auditor predictions on a CSV.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--model-dir", default="models", help="Directory containing trained artifacts")
    parser.add_argument("--output", default="predictions_with_dossiers.csv", help="Output CSV path")
    parser.add_argument("--dossiers", default="dossiers.json", help="Output JSON dossier path")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    pseudo = joblib.load(model_dir / "pseudo_label_artifacts.joblib")
    classifier_artifacts = joblib.load(model_dir / "classifier_artifacts.joblib")

    df = load_and_prepare_for_prediction(args.input)
    df = apply_pseudo_labeler(df, pseudo)
    x = build_classifier_features(df, classifier_artifacts)
    probs = classifier_artifacts["classifier"].predict_proba(x)[:, 1]
    preds = (probs >= classifier_artifacts["threshold"]).astype(int)

    dossiers = []
    for i, row in df.iterrows():
        if preds[i] == 1 and int(row["severity_delta"]) != 0:
            dossiers.append(make_dossier(row, probs[i], pseudo["keywords"]))

    output_df = df[[
        "Ticket_ID",
        "Priority_Level",
        "Ticket_Channel",
        "Issue_Category",
        "Resolution_Time_Hours",
        "inferred_severity_ordinal",
        "severity_delta",
    ]].copy()
    output_df["predicted_mismatch"] = preds
    output_df["mismatch_probability"] = probs
    output_df["inferred_severity"] = output_df["inferred_severity_ordinal"].map(SEVERITY_INV)
    output_df.to_csv(args.output, index=False)

    with open(args.dossiers, "w") as f:
        json.dump(dossiers, f, indent=2)

    print(f"Wrote predictions to {args.output}")
    print(f"Wrote {len(dossiers)} dossiers to {args.dossiers}")


if __name__ == "__main__":
    main()
