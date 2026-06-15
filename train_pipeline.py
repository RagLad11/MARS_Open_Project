import argparse
import json
import os
import pickle
import re
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


SEED = 42
SEVERITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
SEVERITY_INV = {v: k for k, v in SEVERITY_MAP.items()}

LEAKAGE_COLUMNS = [
    "inferred_severity_ordinal",
    "severity_delta",
    "fused_severity_score",
    "sig_a_severity",
    "sig_b_rt_proxy",
    "sig_c_rules",
    "sig_d_cluster",
    "signal_agreement",
    "mismatch_label",
]


def canonicalize_col(col):
    col = str(col).strip().replace("-", " ").replace("/", " ")
    col = "_".join(col.split())
    aliases = {
        "Ticket_ID": "Ticket_ID",
        "Ticket_Subject": "Ticket_Subject",
        "Ticket_Description": "Ticket_Description",
        "Customer_Email": "Customer_Email",
        "Product_Purchased": "Product_Purchased",
        "Ticket_Priority": "Priority_Level",
        "Priority": "Priority_Level",
        "Priority_Level": "Priority_Level",
        "Ticket_Channel": "Ticket_Channel",
        "Resolution_Time": "Resolution_Time_Hours",
        "Resolution_Time_Hours": "Resolution_Time_Hours",
        "Ticket_Type": "Issue_Category",
        "Issue_Category": "Issue_Category",
    }
    return aliases.get(col, col)


def clean_description(text):
    text = str(text).replace("Hi Support,", "").strip()
    parts = [s for s in text.split(". ") if len(s.split()) >= 3]
    return ". ".join(parts[:3]) or text


def categorize_rt(rt):
    if rt < 24:
        return "<24 hrs"
    if rt <= 72:
        return "24-72 hrs"
    if rt <= 120:
        return "72-120 hrs"
    return ">120 hrs"


def load_and_prepare(path):
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

    priority_aliases = {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "critical": "Critical",
        "urgent": "Critical",
        "normal": "Medium",
    }
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
    return df


def ordinal_from_cont(values, ref):
    q = np.percentile(ref, [25, 50, 75])
    return np.digitize(values, q), q


def ordinal_apply(values, q):
    return np.digitize(values, q)


def build_pseudo_labels(df_train, df_val, df_test, model_dir):
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    anchors_text = {
        3: [
            "complete outage service unavailable cannot access emergency production down",
            "critical security breach data loss data compromised financial impact",
            "payment failure for many users business stopped severe incident",
        ],
        2: [
            "application crashes repeatedly blocking important workflow",
            "major functionality broken cannot complete task needs escalation",
            "data not syncing important customer impact",
        ],
        1: [
            "slow performance intermittent error minor disruption workaround available",
            "feature not working as expected but service usable",
            "single user issue inconvenience",
        ],
        0: [
            "general inquiry question how to use service",
            "feature request suggestion feedback nice to have",
            "billing clarification informational request",
        ],
    }
    anchors = {lv: sbert.encode(texts, convert_to_numpy=True, normalize_embeddings=True) for lv, texts in anchors_text.items()}

    def signal_a(split):
        emb = sbert.encode(
            split["combined_text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=True,
        )
        scores = []
        for e in emb:
            sims = {lv: float(np.max(cosine_similarity(e.reshape(1, -1), anchors[lv])[0])) for lv in anchors}
            exp_s = {lv: np.exp(8 * sims[lv]) for lv in sims}
            scores.append(sum(lv * exp_s[lv] for lv in exp_s) / sum(exp_s.values()))
        return np.array(scores), emb

    a_tr_cont, emb_tr = signal_a(df_train)
    a_va_cont, emb_va = signal_a(df_val)
    a_te_cont, emb_te = signal_a(df_test)
    sig_a_tr, qa = ordinal_from_cont(a_tr_cont, a_tr_cont)
    sig_a_va = ordinal_apply(a_va_cont, qa)
    sig_a_te = ordinal_apply(a_te_cont, qa)

    rt_vec = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    def rt_features(split, fit=False):
        text = rt_vec.fit_transform(split["combined_text"]) if fit else rt_vec.transform(split["combined_text"])
        cats = pd.get_dummies(split[["Ticket_Channel", "Issue_Category", "Product_Name"]].astype(str))
        if fit:
            rt_features.cat_cols = cats.columns
        cats = cats.reindex(columns=rt_features.cat_cols, fill_value=0)
        return sparse.hstack([text, sparse.csr_matrix(cats.values)], format="csr")

    x_rt_tr = rt_features(df_train, True)
    x_rt_va = rt_features(df_val, False)
    x_rt_te = rt_features(df_test, False)
    rt_model = xgb.XGBRegressor(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
    )
    rt_model.fit(x_rt_tr, df_train["Resolution_Time_Hours"].values)
    rt_tr = rt_model.predict(x_rt_tr)
    rt_va = rt_model.predict(x_rt_va)
    rt_te = rt_model.predict(x_rt_te)
    sig_b_tr, qb = ordinal_from_cont(rt_tr, rt_tr)
    sig_b_va = ordinal_apply(rt_va, qb)
    sig_b_te = ordinal_apply(rt_te, qb)

    critical_kws = ["outage", "production down", "security breach", "data loss", "data lost", "breach", "cannot access"]
    high_kws = ["cannot login", "crash", "crashes", "blocked", "blocking", "not syncing", "urgent", "escalate"]
    esc_kws = ["ceo", "manager", "legal", "complaint", "sla", "enterprise"]
    low_kws = ["question", "how do i", "feature request", "suggestion", "feedback", "clarification"]
    neg_kws = ["not urgent", "no outage", "false alarm", "resolved", "workaround available", "can wait"]

    def signal_c(split):
        out = []
        for text in split["combined_text"].str.lower():
            score = 1
            if any(k in text for k in low_kws):
                score -= 1
            if any(k in text for k in high_kws):
                score += 1
            if any(k in text for k in esc_kws):
                score += 1
            if any(k in text for k in critical_kws):
                score += 2
            if any(k in text for k in neg_kws):
                score -= 2
            out.append(int(np.clip(score, 0, 3)))
        return np.array(out)

    sig_c_tr, sig_c_va, sig_c_te = signal_c(df_train), signal_c(df_val), signal_c(df_test)

    kmeans = KMeans(n_clusters=10, random_state=SEED, n_init=10)
    cl_tr = kmeans.fit_predict(emb_tr)
    rt_norm = 3 * np.clip((rt_tr - np.percentile(rt_tr, 1)) / (np.percentile(rt_tr, 99) - np.percentile(rt_tr, 1) + 1e-9), 0, 1)
    cluster_score = {}
    for c in range(10):
        mask = cl_tr == c
        cluster_score[c] = np.mean(0.65 * a_tr_cont[mask] + 0.35 * rt_norm[mask]) if mask.sum() else np.mean(a_tr_cont)

    d_tr_cont = np.array([cluster_score[c] for c in cl_tr])
    d_va_cont = np.array([cluster_score[c] for c in kmeans.predict(emb_va)])
    d_te_cont = np.array([cluster_score[c] for c in kmeans.predict(emb_te)])
    sig_d_tr, qd = ordinal_from_cont(d_tr_cont, d_tr_cont)
    sig_d_va = ordinal_apply(d_va_cont, qd)
    sig_d_te = ordinal_apply(d_te_cont, qd)

    best_weights, best_score = None, -1
    for weights in product(np.linspace(0.05, 0.70, 14), repeat=4):
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            continue
        fused = sum(w * s for w, s in zip(weights, [sig_a_tr, sig_b_tr, sig_c_tr, sig_d_tr]))
        fused_disc, _ = ordinal_from_cont(fused, fused)
        kappas = [cohen_kappa_score(fused_disc, s) for s in [sig_a_tr, sig_b_tr, sig_c_tr, sig_d_tr]]
        balance = 1 - np.std(np.bincount(fused_disc, minlength=4) / len(fused_disc))
        score = np.mean(kappas) + 0.10 * balance
        if score > best_score:
            best_score, best_weights = score, weights

    fused_tr = sum(w * s for w, s in zip(best_weights, [sig_a_tr, sig_b_tr, sig_c_tr, sig_d_tr]))
    qf = np.percentile(fused_tr, [25, 50, 75])
    splits = [
        (df_train, [sig_a_tr, sig_b_tr, sig_c_tr, sig_d_tr], fused_tr),
        (df_val, [sig_a_va, sig_b_va, sig_c_va, sig_d_va], sum(w * s for w, s in zip(best_weights, [sig_a_va, sig_b_va, sig_c_va, sig_d_va]))),
        (df_test, [sig_a_te, sig_b_te, sig_c_te, sig_d_te], sum(w * s for w, s in zip(best_weights, [sig_a_te, sig_b_te, sig_c_te, sig_d_te]))),
    ]

    for split, sigs, fused in splits:
        inferred = ordinal_apply(fused, qf)
        split["assigned_ordinal"] = split["Priority_Level"].map(SEVERITY_MAP).astype(int)
        split["inferred_severity_ordinal"] = inferred
        split["fused_severity_score"] = fused
        split["severity_delta"] = inferred - split["assigned_ordinal"]
        split["mismatch_label"] = (np.abs(split["severity_delta"]) >= 1).astype(int)
        split["sig_a_severity"], split["sig_b_rt_proxy"], split["sig_c_rules"], split["sig_d_cluster"] = sigs
        split["signal_agreement"] = 1.0 - (np.std(np.column_stack(sigs), axis=1) / 1.5).clip(0, 1)

    pseudo_artifacts = {
        "sbert_name": "all-MiniLM-L6-v2",
        "anchors_text": anchors_text,
        "anchor_embeddings": anchors,
        "qa": qa,
        "qb": qb,
        "qd": qd,
        "qf": qf,
        "rt_vectorizer": rt_vec,
        "rt_cat_cols": rt_features.cat_cols,
        "rt_model": rt_model,
        "kmeans": kmeans,
        "cluster_score": cluster_score,
        "best_weights": best_weights,
        "keywords": {
            "critical": critical_kws,
            "high": high_kws,
            "escalation": esc_kws,
            "low": low_kws,
            "negation": neg_kws,
        },
    }
    joblib.dump(pseudo_artifacts, model_dir / "pseudo_label_artifacts.joblib")
    return df_train, df_val, df_test, pseudo_artifacts


def build_classifier_features(df, vectorizer=None, cat_cols=None, fit=False):
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
    if fit:
        vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2), min_df=3, max_df=0.92, sublinear_tf=True)
        text_x = vectorizer.fit_transform(text)
    else:
        text_x = vectorizer.transform(text)

    cats = pd.get_dummies(df[["Priority_Level", "Ticket_Channel", "Issue_Category", "RT_Bucket"]].astype(str))
    if fit:
        cat_cols = cats.columns
    cats = cats.reindex(columns=cat_cols, fill_value=0)
    nums = df[["Resolution_Time_Hours", "assigned_ordinal"]].astype(float).values
    x = sparse.hstack([text_x, sparse.csr_matrix(cats.values), sparse.csr_matrix(nums)], format="csr")
    return x, vectorizer, cat_cols


def train_classifier(df_train, df_val, df_test, model_dir):
    y_train = df_train["mismatch_label"].values
    y_val = df_val["mismatch_label"].values
    y_test = df_test["mismatch_label"].values

    x_train, vec, cat_cols = build_classifier_features(df_train, fit=True)
    x_val, _, _ = build_classifier_features(df_val, vec, cat_cols)
    x_test, _, _ = build_classifier_features(df_test, vec, cat_cols)

    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=2,
        min_child_weight=8,
        learning_rate=0.035,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=8.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(x_train, y_train)

    val_probs = clf.predict_proba(x_val)[:, 1]
    best = {"score": -1, "threshold": 0.5}
    rows = []
    for th in np.arange(0.10, 0.91, 0.01):
        pred = (val_probs >= th).astype(int)
        acc = accuracy_score(y_val, pred)
        macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        r0 = recall_score(y_val, pred, pos_label=0, zero_division=0)
        r1 = recall_score(y_val, pred, pos_label=1, zero_division=0)
        score = macro_f1 + 0.05 * min(r0, r1)
        rows.append({"threshold": float(th), "accuracy": acc, "macro_f1": macro_f1, "recall_0": r0, "recall_1": r1})
        if score > best["score"]:
            best = {"score": score, "threshold": float(th)}

    test_probs = clf.predict_proba(x_test)[:, 1]
    test_pred = (test_probs >= best["threshold"]).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "macro_f1": float(f1_score(y_test, test_pred, average="macro", zero_division=0)),
        "recall_0": float(recall_score(y_test, test_pred, pos_label=0, zero_division=0)),
        "recall_1": float(recall_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "threshold": best["threshold"],
        "leakage_columns_excluded": LEAKAGE_COLUMNS,
    }

    joblib.dump(
        {
            "classifier": clf,
            "vectorizer": vec,
            "cat_cols": cat_cols,
            "threshold": best["threshold"],
            "severity_map": SEVERITY_MAP,
        },
        model_dir / "classifier_artifacts.joblib",
    )
    pd.DataFrame(rows).to_csv(model_dir / "threshold_search.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the Support Integrity Auditor pipeline.")
    parser.add_argument("--data", required=True, help="Path to customer_support_tickets.csv")
    parser.add_argument("--model-dir", default="models", help="Directory for trained artifacts")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory for reports")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    outputs_dir = Path(args.outputs_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(args.data)
    df_train_full, df_test = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["Priority_Level"])
    df_train, df_val = train_test_split(
        df_train_full, test_size=0.15 / 0.85, random_state=SEED, stratify=df_train_full["Priority_Level"]
    )

    df_train, df_val, df_test, pseudo = build_pseudo_labels(df_train.copy(), df_val.copy(), df_test.copy(), model_dir)
    metrics = train_classifier(df_train, df_val, df_test, model_dir)

    df_train.to_csv(outputs_dir / "train_pseudo_labeled.csv", index=False)
    df_val.to_csv(outputs_dir / "val_pseudo_labeled.csv", index=False)
    df_test.to_csv(outputs_dir / "test_pseudo_labeled.csv", index=False)

    signal_cols = ["sig_a_severity", "sig_b_rt_proxy", "sig_c_rules", "sig_d_cluster"]
    agreement = {}
    for i, a in enumerate(signal_cols):
        for b in signal_cols[i + 1 :]:
            agreement[f"{a}_vs_{b}"] = float(cohen_kappa_score(df_train[a], df_train[b]))

    report = {
        "metrics": metrics,
        "fusion_weights": dict(zip(["SBERT", "resolution_proxy", "rules", "clusters"], map(float, pseudo["best_weights"]))),
        "pseudo_label_rate_train": float(df_train["mismatch_label"].mean()),
        "pairwise_signal_agreement_train": agreement,
        "anti_overfit_note": "Classifier features exclude pseudo-label answer columns and post-hoc/high-cardinality fields.",
    }
    with open(outputs_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
