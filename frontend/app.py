"""PTB-XL ECG inference frontend - Streamlit multi-page app."""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

# Read the backend URL from the environment so it works inside Docker Compose. 
# Trailing slash stripped to avoid double-slash in URLs like http://backend:8000//predict.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000").rstrip("/")

# The 5 PTB-XL superclasses in the exact order the model outputs them.
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _classes_to_labels(classes: list[str]) -> list[int]:
    """Convert active class names to a binary vector e.g. ["NORM","STTC"] -> [1,0,1,0,0]."""
    return [1 if cls in classes else 0 for cls in SUPERCLASSES]


def _labels_to_classes(labels: list[int]) -> list[str]:
    """Convert a binary vector to active class names e.g. [1,0,1,0,0] -> ["NORM","STTC"]."""
    return [cls for cls, v in zip(SUPERCLASSES, labels) if v]


def _normalize(pred: dict) -> dict:
    """Translate a backend prediction record to the shape the frontend uses internally.

    The backend uses predicted_class/confirmed_class (lists of strings);
    the frontend works with predicted_labels/confirmed_labels (binary vectors).
    """
    pred["predicted_labels"] = _classes_to_labels(pred.get("predicted_class", []))
    pred["confirmed_labels"] = _classes_to_labels(pred.get("confirmed_class") or [])
    pred.setdefault("confirmed_at", None)
    return pred

# Full human-readable names for each superclass, used in the label reference table.
SUPERCLASS_NAMES = {
    "NORM": "Normal",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T-wave Changes",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}

# Maps the model identifier string (sent to the backend) to a display label.
# These exact keys are what the backend's `model` field must accept.
MODELS = {
    "bilstm_attention":  "BiLSTM Attention ★ Best BiLSTM",
    "bilstm_regularized": "BiLSTM Regularized",
    "bilstm_baseline":   "BiLSTM Baseline",
    "bilstm_smaller":    "BiLSTM Smaller",
    "cnn_baseline":      "1D CNN Baseline",
    "cnn_small":         "1D CNN Small ★ Best CNN",
}

# Static metadata for each model, used on the Model Info page.
# Metrics come from Week 3 evaluation on the PTB-XL test fold.
MODEL_INFO = {
    "bilstm_attention": {
        "architecture": "Bidirectional LSTM with attention mechanism",
        "file": "bilstm_attention.pt",
        "role": "Best BiLSTM - highest macro AUC",
        "macro_auc": 0.9255,
        "macro_f1": 0.7292,
        "per_class_auc": {"NORM": 0.9433, "MI": 0.9305, "STTC": 0.9335, "CD": 0.9139, "HYP": 0.9065},
    },
    "bilstm_regularized": {
        "architecture": "Bidirectional LSTM with L2 regularization",
        "file": "bilstm_regularized.pt",
        "role": "Experiment - regularization variant (degraded performance)",
        "macro_auc": 0.6585,
        "macro_f1": 0.4598,
        "per_class_auc": {"NORM": 0.6937, "MI": 0.6291, "STTC": 0.6358, "CD": 0.6493, "HYP": 0.6845},
    },
    "bilstm_baseline": {
        "architecture": "Bidirectional LSTM baseline (no attention)",
        "file": "bilstm_baseline.pt",
        "role": "Experiment - baseline",
        "macro_auc": 0.9084,
        "macro_f1": 0.7158,
        "per_class_auc": {"NORM": 0.9360, "MI": 0.9035, "STTC": 0.9198, "CD": 0.9013, "HYP": 0.8812},
    },
    "bilstm_smaller": {
        "architecture": "Smaller bidirectional LSTM (fewer hidden units)",
        "file": "bilstm_smaller.pt",
        "role": "Experiment - lightweight variant",
        "macro_auc": 0.9124,
        "macro_f1": 0.7212,
        "per_class_auc": {"NORM": 0.9373, "MI": 0.9156, "STTC": 0.9218, "CD": 0.8975, "HYP": 0.8899},
    },
    "cnn_baseline": {
        "architecture": "1D CNN - larger filter configuration",
        "file": "cnn_baseline.pth",
        "role": "Experiment - baseline",
        "macro_auc": 0.9066,
        "macro_f1": 0.7052,
        "per_class_auc": {"NORM": 0.9340, "MI": 0.8917, "STTC": 0.9243, "CD": 0.8991, "HYP": 0.8840},
    },
    "cnn_small": {
        "architecture": "1D CNN - [32, 64, 128] filter channels",
        "file": "cnn_small.pth",
        "role": "Best CNN - highest macro AUC",
        "macro_auc": 0.9128,
        "macro_f1": 0.7052,
        "per_class_auc": {"NORM": 0.9356, "MI": 0.8949, "STTC": 0.9278, "CD": 0.9093, "HYP": 0.8967},
    },
}


# ── API helper ────────────────────────────────────────────────────────────────


def _api(method: str, path: str, **kwargs):
    """Make an HTTP request to the backend and return (data, error).

    On success returns (parsed JSON, None).
    On any failure returns (None, human-readable error string) so the caller
    can show it with st.error() without crashing the app.
    """
    try:
        r = requests.request(method, f"{BACKEND_URL}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        # Backend is not reachable at all (e.g. server not started).
        return None, f"Cannot connect to backend at **{BACKEND_URL}**. Is the server running?"
    except requests.exceptions.HTTPError as e:
        # Backend responded with a 4xx/5xx. Try to extract FastAPI's detail message.
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        return None, f"Backend returned {e.response.status_code}: {detail}"
    except Exception as e:
        return None, str(e)


# ── Live model list helper ────────────────────────────────────────────────────


def _fetch_available_models() -> list[str]:
    """Return model keys that are currently ready on the backend.

    Falls back to the full hardcoded list if the backend is unreachable,
    so the page still works against the mock server.
    """
    data, _ = _api("GET", "/models")
    if data:
        # Backend returns [{"name": "bilstm_attention"}, ...] — filter to known keys only.
        live = [m["name"] for m in data if m["name"] in MODELS]
        if live:
            return live
    return list(MODELS.keys())


# ── Page 1: New Prediction ────────────────────────────────────────────────────


def page_new_prediction():
    st.title("New Prediction")
    st.caption(
        "Upload a ZIP archive containing a PTB-XL ECG record (.dat + .hea pair) "
        "and select a model to run multi-label superclass classification."
    )

    # Single ZIP uploader - the backend unpacks the .dat and .hea files itself.
    zip_file = st.file_uploader(
        "ECG record ZIP (must contain the .dat + .hea pair)",
        type=["zip"],
    )

    # Radio to choose between single-model and ensemble mode.
    mode = st.radio(
        "Inference mode",
        ["Single model", "Ensemble (all models)"],
        horizontal=True,
    )

    model_key = None
    if mode == "Single model":
        # Dropdown populated from the live backend model list.
        # Models that are currently retraining are excluded by the backend automatically.
        available = _fetch_available_models()
        model_key = st.selectbox(
            "Model",
            options=available,
            format_func=lambda k: MODELS.get(k, k),
        )
    else:
        st.caption("All available models will run and their logits averaged into one prediction.")

    if not zip_file:
        st.info("Upload a ZIP archive containing the .dat and .hea files to enable prediction.")

    if st.button("Predict", type="primary", disabled=not zip_file):
        with st.spinner("Running inference…"):
            if mode == "Single model":
                data, err = _api(
                    "POST",
                    "/predict",
                    files={"file": (zip_file.name, zip_file.getvalue(), "application/zip")},
                    data={"model_name": model_key},
                )
                if not err:
                    data["model"] = data.get("model_name", model_key)
                    data["individual_predictions"] = None
            else:
                data, err = _api(
                    "POST",
                    "/predict_all",
                    files={"file": (zip_file.name, zip_file.getvalue(), "application/zip")},
                )
                if not err:
                    data["model"] = data.get("model_name", "ensemble")

        if err:
            st.error(err)
        else:
            data = _normalize(data)
            st.session_state["last_prediction"] = data
            # Pre-tick the confirmation checkboxes to match the prediction.
            for i, cls in enumerate(SUPERCLASSES):
                st.session_state[f"confirm_{cls}"] = bool(data["predicted_labels"][i])

    # Nothing to show until at least one prediction has been made.
    pred = st.session_state.get("last_prediction")
    if pred is None:
        return

    st.divider()

    # ── Results ───────────────────────────────────────────────────────────────

    labels = pred["predicted_labels"]
    probs = pred["probabilities"]
    active = [SUPERCLASSES[i] for i, v in enumerate(labels) if v == 1]

    st.subheader("Results")
    if pred.get("model") == "ensemble":
        st.caption("Ensemble prediction — averaged logits from all available models")
    else:
        st.caption(f"Model: **{MODELS.get(pred.get('model'), pred.get('model', ''))}**")

    if active:
        st.success(f"Predicted active: **{', '.join(active)}**")
    else:
        st.warning("No labels predicted active at threshold 0.5")

    st.write("**Confidence per class:** " + "  ·  ".join(f"{cls}: {probs[cls]:.1%}" for cls in SUPERCLASSES))

    chart_df = pd.DataFrame(
        {"Probability": [probs[cls] for cls in SUPERCLASSES]},
        index=SUPERCLASSES,
    )
    st.bar_chart(chart_df, color="#800000")

    # For ensemble predictions show the per-model breakdown in an expander.
    individual = pred.get("individual_predictions")
    if individual:
        with st.expander("Per-model breakdown"):
            for m_name, m_pred in individual.items():
                m_active = " ".join(m_pred["predicted_class"]) or "-"
                st.write(f"**{MODELS.get(m_name, m_name)}:** {m_active}")
                m_chart = pd.DataFrame(
                    {"Probability": [m_pred["probabilities"][cls] for cls in SUPERCLASSES]},
                    index=SUPERCLASSES,
                )
                st.bar_chart(m_chart, color="#800000")

    with st.expander("Raw binary vector"):
        st.code(f"[{', '.join(str(v) for v in labels)}]  # NORM  MI  STTC  CD  HYP")

    # ── Confirm / correct ─────────────────────────────────────────────────────

    st.divider()
    st.subheader("Confirm / correct labels")
    st.caption("Adjust the checkboxes if the prediction is wrong, then click Confirm.")

    cols = st.columns(len(SUPERCLASSES))
    checked: dict[str, bool] = {}
    for i, cls in enumerate(SUPERCLASSES):
        with cols[i]:
            checked[cls] = st.checkbox(cls, key=f"confirm_{cls}")

    notes = st.text_input("Notes (optional)")

    if st.button("Confirm", type="secondary"):
        confirmed_class = _labels_to_classes([1 if checked[cls] else 0 for cls in SUPERCLASSES])
        data, err = _api(
            "POST",
            f"/predictions/{pred['prediction_id']}/confirm",
            json={"confirmed_class": confirmed_class, "notes": notes or None},
        )
        if err:
            st.error(err)
        else:
            data["model"] = pred.get("model")
            data = _normalize(data)
            st.session_state["last_prediction"] = data
            st.success("Labels confirmed.")


# ── Page 2: Prediction History ────────────────────────────────────────────────


def page_history():
    st.title("Prediction History")

    history_msg = st.session_state.pop("history_confirm_success", None)
    if history_msg:
        st.success(history_msg)

    # Fetch the 20 most recent prediction records from the backend.
    # The backend returns a paginated envelope {total, limit, offset, results: [...]}.
    raw, err = _api("GET", "/predictions?limit=20&offset=0")
    if err:
        st.error(err)
        return
    data = [_normalize(p) for p in (raw.get("results") if isinstance(raw, dict) else raw) or []]
    if not data:
        st.info("No predictions yet. Run some on the New Prediction page.")
        return

    # ── Filters ───────────────────────────────────────────────────────────────

    col1, col2 = st.columns(2)
    with col1:
        confirmed_filter = st.selectbox("Status", ["All", "Confirmed", "Unconfirmed"])
    with col2:
        family_filter = st.selectbox("Model family", ["All", "BiLSTM only", "CNN only"])

    # ── Build display dataframe ───────────────────────────────────────────────

    # Flatten each prediction record into a display-friendly row.
    rows = []
    for p in data:
        active_str = " ".join(p.get("predicted_class", [])) or "-"
        rows.append(
            {
                "Time": p["created_at"].replace("T", " ").split(".")[0],
                "Model": MODELS.get(p.get("model_name"), p.get("model_name", "Unknown")),
                "Predicted": active_str,
                # Confidence shown as the highest probability across all classes.
                "Confidence": f"{max(p['probabilities'].values()):.0%}",
                "Confirmed labels": " ".join(p.get("confirmed_class") or []) or "-",
                "Status": "✓ Confirmed" if p["confirmed"] else "Pending",
                # Hidden column - used to look up the full record when a row is clicked.
                "_id": p["prediction_id"],
            }
        )

    df = pd.DataFrame(rows)

    # Apply the confirmed/unconfirmed filter.
    if confirmed_filter == "Confirmed":
        df = df[df["Status"] == "✓ Confirmed"]
    elif confirmed_filter == "Unconfirmed":
        df = df[df["Status"] == "Pending"]

    # Apply the model family filter by matching display label substrings.
    # Note: model name is not stored in history records by the backend, so
    # records without a known model are excluded from family filters.
    if family_filter == "BiLSTM only":
        df = df[df["Model"].str.contains("BiLSTM", na=False)]
    elif family_filter == "CNN only":
        df = df[df["Model"].str.contains("CNN|ensemble", na=False)]

    if df.empty:
        st.info("No records match the current filters.")
        return

    display_cols = ["Time", "Model", "Predicted", "Confidence", "Confirmed labels", "Status"]
    # on_select="rerun" makes the table interactive - clicking a row triggers
    # a Streamlit rerun and we can read which row was selected below.
    event = st.dataframe(
        df[display_cols].reset_index(drop=True),
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # ── Row detail ────────────────────────────────────────────────────────────

    selected_rows = event.selection.rows if event else []
    if not selected_rows:
        st.caption("Click a row to view the full record.")
        return

    # Look up the full record by ID using the hidden _id column.
    selected_id = df.iloc[selected_rows[0]]["_id"]
    detail, err = _api("GET", f"/predictions/{selected_id}")
    if err:
        st.error(err)
        return
    detail = _normalize(detail)

    st.divider()
    st.subheader("Record detail")

    # Split the detail into two columns for a compact layout.
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**ID:** `{detail['prediction_id']}`")
        st.write(f"**Created:** {detail['created_at']}")
        active_str = " ".join(detail.get("predicted_class", [])) or "-"
        st.write(f"**Predicted:** {active_str}")
    with c2:
        st.write(f"**Confirmed:** {'Yes' if detail['confirmed'] else 'No'}")
        if detail["confirmed"]:
            conf_str = " ".join(detail.get("confirmed_class") or []) or "-"
            st.write(f"**Confirmed labels:** {conf_str}")
            if detail.get("confirmed_at"):
                st.write(f"**Confirmed at:** {detail['confirmed_at']}")
        if detail.get("notes"):
            st.write(f"**Notes:** {detail['notes']}")

    # Bar chart of probabilities for this historical record.
    chart_df = pd.DataFrame(
        {"Probability": [detail["probabilities"][cls] for cls in SUPERCLASSES]},
        index=SUPERCLASSES,
    )
    st.bar_chart(chart_df, color="#800000")

    st.divider()
    st.subheader("Confirm / correct labels")
    st.caption("Select the true classes for this record and save them.")

    key_prefix = f"history_confirm_{detail['prediction_id']}_"
    init_key = f"{key_prefix}initialized"
    if not st.session_state.get(init_key):
        preselected = detail.get("confirmed_class") or detail.get("predicted_class") or []
        for cls in SUPERCLASSES:
            st.session_state[f"{key_prefix}{cls}"] = cls in preselected
        st.session_state[f"{key_prefix}notes"] = detail.get("notes") or ""
        st.session_state[init_key] = True

    cols = st.columns(len(SUPERCLASSES))
    checked: dict[str, bool] = {}
    for i, cls in enumerate(SUPERCLASSES):
        with cols[i]:
            checked[cls] = st.checkbox(cls, key=f"{key_prefix}{cls}")

    notes = st.text_input("Notes (optional)", key=f"{key_prefix}notes")

    if st.button("Save confirmation", type="secondary", key=f"{key_prefix}submit"):
        confirmed_class = _labels_to_classes([1 if checked[cls] else 0 for cls in SUPERCLASSES])
        _, err = _api(
            "POST",
            f"/predictions/{detail['prediction_id']}/confirm",
            json={"confirmed_class": confirmed_class, "notes": notes or None},
        )
        if err:
            st.error(err)
        else:
            st.session_state[init_key] = False
            st.session_state["history_confirm_success"] = "Labels confirmed."
            st.rerun()


# ── Page 3: Model Info ────────────────────────────────────────────────────────


def page_model_info():
    st.title("Model Info")

    # Three tabs: browse individual models, compare the two best, reference table.
    tab_browse, tab_compare, tab_labels = st.tabs(["Browse Models", "Comparison", "Label Reference"])

    # ── Browse ────────────────────────────────────────────────────────────────

    with tab_browse:
        # Dropdown to pick which model to inspect.
        selected_key = st.selectbox(
            "Model",
            options=list(MODEL_INFO.keys()),
            format_func=lambda k: MODELS[k],
        )
        info = MODEL_INFO[selected_key]

        st.write(f"**Architecture:** {info['architecture']}")
        st.write(f"**Model file:** `{info['file']}`")
        st.write(f"**Role:** {info['role']}")

        if info["macro_auc"] is not None:
            st.divider()
            m1, m2 = st.columns(2)
            m1.metric("Macro AUC (test)", f"{info['macro_auc']:.4f}")
            m2.metric("Macro F1 (test)", f"{info['macro_f1']:.4f}")

            if info["per_class_auc"]:
                st.write("**Per-class AUC (test fold)**")
                auc_df = pd.DataFrame(
                    {"AUC": list(info["per_class_auc"].values())},
                    index=list(info["per_class_auc"].keys()),
                )
                st.bar_chart(auc_df, color="#800000")
        else:
            st.caption("Detailed metrics not recorded for this variant (experiment run only).")

    # ── Comparison ────────────────────────────────────────────────────────────

    with tab_compare:
        st.subheader("Best BiLSTM vs Best CNN")

        best_bilstm = MODEL_INFO["bilstm_attention"]
        best_cnn = MODEL_INFO["cnn_small"]

        # Build a side-by-side comparison dict - one key per model, rows are metrics.
        comp = {
            "BiLSTM Attention": {
                "Dataset": "PTB-XL 100 Hz",
                "Model file": best_bilstm["file"],
                "Macro AUC": f"{best_bilstm['macro_auc']:.4f}",
                "Macro F1": f"{best_bilstm['macro_f1']:.4f}",
                **{f"AUC - {cls}": f"{best_bilstm['per_class_auc'][cls]:.4f}" for cls in SUPERCLASSES},
            },
            "1D CNN Small": {
                "Dataset": "PTB-XL 100 Hz",
                "Model file": best_cnn["file"],
                "Macro AUC": f"{best_cnn['macro_auc']:.4f}",
                "Macro F1": f"{best_cnn['macro_f1']:.4f}",
                **{f"AUC - {cls}": f"{best_cnn['per_class_auc'][cls]:.4f}" for cls in SUPERCLASSES},
            },
        }
        st.dataframe(pd.DataFrame(comp), use_container_width=True)

        st.info(
            "**Week 3 finding:** BiLSTM with attention outperforms 1D CNN on PTB-XL ECG "
            "superclass classification (macro AUC 0.9255 vs 0.9145). The attention mechanism "
            "lets the model focus on diagnostically relevant time windows in the signal."
        )

    # ── Label reference ───────────────────────────────────────────────────────

    with tab_labels:
        st.subheader("PTB-XL superclass label mapping")
        st.caption(
            "The predicted_labels vector has 5 elements in this exact order. "
            "Multiple labels can be active simultaneously (multi-label classification)."
        )
        # Table mapping vector index -> short key -> full clinical name.
        label_df = pd.DataFrame(
            [
                {"Index": i, "Key": cls, "Full name": SUPERCLASS_NAMES[cls]}
                for i, cls in enumerate(SUPERCLASSES)
            ]
        )
        st.dataframe(label_df, use_container_width=True, hide_index=True)

        st.write("**Example**")
        st.code("[1, 0, 1, 0, 0]  ->  NORM active, STTC active, MI/CD/HYP inactive")


# ── Page 4: Retrain ───────────────────────────────────────────────────────────


def page_retrain():
    st.title("Retrain Model")
    st.caption(
        "Trigger background retraining on confirmed predictions stored in the database. "
        "The model stays available for inference during retraining and is updated when done."
    )

    st.info(
        "Retraining uses all confirmed predictions as training data. "
        "You need at least a few confirmed examples before this will work — "
        "go to New Prediction and confirm some results first."
    )

    # Fetch which models are currently ready (retraining ones are excluded).
    available = _fetch_available_models()
    all_model_keys = list(MODELS.keys())

    # Show all known models with their current status.
    st.subheader("Model status")
    status_rows = []
    for key in all_model_keys:
        status_rows.append({
            "Model": MODELS.get(key, key),
            "Status": "Ready" if key in available else "Retraining...",
        })
    st.dataframe(
        status_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Start retraining")

    if not available:
        st.warning("No models are currently available — all may be retraining already.")
        return

    model_key = st.selectbox(
        "Select model to retrain",
        options=available,
        format_func=lambda k: MODELS.get(k, k),
    )

    st.caption(
        "Retraining runs in the background. The model will not appear as 'Retraining' "
        "immediately — refresh this page after a few seconds to see the updated status."
    )

    if st.button("Start Retraining", type="primary"):
        data, err = _api("POST", f"/retrain/{model_key}")
        if err:
            st.error(err)
        else:
            st.success(
                f"Retraining started for **{MODELS.get(model_key, model_key)}**. "
                "Refresh this page to track status."
            )


# ── App entry point ───────────────────────────────────────────────────────────


st.set_page_config(
    page_title="PTB-XL ECG Inference",
    page_icon=None,
    layout="wide",
)

# Custom CSS to apply the maroon (#800000) and white colour scheme.
# Streamlit's config.toml handles widget-level theming (checkboxes, sliders, etc.);
# this block covers elements config.toml cannot reach (sidebar, buttons, table headers).
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #800000;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    /* Active nav item */
    [data-testid="stSidebarNavLink"][aria-selected="true"] {
        background-color: rgba(0,0,0,0.12) !important;
        font-weight: 700;
    }
    /* Primary buttons */
    div.stButton > button[kind="primary"] {
        background-color: #800000;
        color: #FFFFFF;
        border: none;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #990000;
        color: #FFFFFF;
    }
    /* Metric value */
    [data-testid="stMetricValue"] {
        color: #990000;
    }
    /* Dataframe header */
    .stDataFrame thead th {
        background-color: #800000 !important;
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Register the three pages and launch the app.
pg = st.navigation(
    [
        st.Page(page_new_prediction, title="New Prediction", default=True),
        st.Page(page_history, title="Prediction History"),
        st.Page(page_model_info, title="Model Info"),
        st.Page(page_retrain, title="Retrain"),
    ]
)
pg.run()
