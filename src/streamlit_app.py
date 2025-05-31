# Import Libraries
import time
import joblib
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import ConfusionMatrixDisplay


@st.cache_resource(show_spinner="Loading model and data...")
# Load model
def load_artifacts():
    model = joblib.load("models/Final XGB + SMOTE Tuned Model.pkl")
    # Data
    X_train, X_test, y_train, y_test = joblib.load(
        "data/processed/split_data.pkl")

    # Cache
    with open("data/processed/profit_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    return model, X_test, y_test, cache


model, X_test, y_test, cache = load_artifacts()
pen_grid = np.array(cache["pen_grid"])

# Main container CSS
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {
    max-width: 100%;
    padding-left: 2rem;
    padding-right: 2rem;
}

.block-container {
    max-width: 100% !important;
    margin-left: 0rem !important;
    margin-right: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}

.css-1kyxreq, .stSelectbox {
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# LIME Section CSS
st.markdown("""
<style>
[data-testid="stExpander"] label,
[data-baseweb="select"]      label {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}

details summary {
    font-size: 1.20rem !important;
}

.col-info-wrap {
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%;
}
.fact-label  {font-size: 1.20rem !important;}
.fact-body   {font-size: 1.05rem !important;}
</style>
""", unsafe_allow_html=True)

# Live inference button CSS
st.markdown("""
<style>
[data-testid="stExpander"] .stButton > button{
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    background: #be123c !important;
    color: #ffffff !important;
    padding: 0.55rem 1.4rem !important;
    border: 1px solid #be123c !important;
    border-radius: 6px !important;
    transition: background 120ms ease-in;
}
[data-testid="stExpander"] .stButton > button:hover{
    background: #9f1239 !important;
    border-color: #9f1239 !important;
}
</style>
""", unsafe_allow_html=True)

# KPI badge CSS
st.markdown("""
<style>
.badge-profit{
    display:inline-block;
    padding:4px 14px;
    border-radius:8px;
    border:2px solid currentColor;
    background-color:rgba(255,255,255,0.04);
}
</style>
""", unsafe_allow_html=True)

# Sidebar business sliders
st.sidebar.header("Business Scenario")
st.sidebar.markdown(
    (
        "Simulating the **annual profits of a fraud detection company** "
        "using the **XGBoost + SMOTE** machine learning model.\n\n"
        "- Money earned for **successfully catching fraud**.\n"
        "- Money lost for **missing fraud** (the bank eats the loss).\n"
        "- Money lost for **wrongly blocking legit customer transactions**\n"
        "  (review cost).\n\n"
        "Adjust the sliders to explore profit generated "
        "across the annual **83 billion card transactions in Europe**."
    ),
    unsafe_allow_html=True
)
st.sidebar.header("Business Assumptions")

avg_fraud = st.sidebar.slider(
    "Average fraud amount ($)",
    100, 140, 120, step=20,
    help="Typical $ value of a confirmed fraudulent transaction"
)
tp_fee_pct = st.sidebar.slider(
    "Reward for catching fraud (% of fraud amount)", 5, 15, 10, step=5,
    help="Success-fee your bank collects when you block a fraud"
) / 100
fp_cost = st.sidebar.slider(
    "Cost of wrongly blocking a customer transaction ($)",
    2.00, 5.00, 3.00, step=1.00,
    help="Customer service / UX cost when you wrongly flag a legit purchase"
)
fn_penalty_pct = st.sidebar.slider(
    "Penalty for missing fraud (% of fraud amount)", 0, 50, 25, step=25,
    help="Loss borne when a fraud goes undetected"
) / 100

# Cache lookup key
cache_lookup_key = (avg_fraud, int(tp_fee_pct * 100), round(fp_cost, 2),
                    int(fn_penalty_pct * 100),)

values = cache[cache_lookup_key]
tau_opt = values["tau_opt"]
profit_opt = values["profit_opt"]
profit_fixed = values["profit_fixed"]
curve_best = np.array(values["curve_best"])
curve_fixed = np.array(values["curve_fixed"])
cm_opt_list = values["cm_opt"]
cm_fixed_list = values["cm_fixed"]
example_idx = values["example_indices"]

cm_opt = np.array(cm_opt_list).reshape(2, 2)
cm_fixed = np.array(cm_fixed_list).reshape(2, 2)

# Page layout
st.title("Machine Learning for Credit Card Fraud-Detection")
st.title("Profit Simulator")
st.markdown(
    """
    <style>
    /* KPI number bigger & green */
    div[data-testid="stMetricValue"] {font-size: 2.1rem; color: #008000;}
    /* delta smaller & grey */
    div[data-testid="stMetricDelta"] {font-size: 0.9rem; color: #666;}
    /* center big title */
    h1 {text-align:center;}
    h2 {text-align:center; margin:0.8rem 0 0.6rem 0;}
    .empty-col div {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Top KPIs
total_tx_eu = 83000000000
kpi_ann = profit_opt / 56000 * total_tx_eu
profit_color_ann = "#008000" if kpi_ann >= 0 else "#FF0000"
profit_color_56k = "#008000" if profit_opt >= 0 else "#FF0000"
delta_val = int(profit_opt - profit_fixed)

# KPI Styling
st.markdown(
    """
    <style>
    .kpi-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 2.6rem;
        font-weight: 700;
    }
    .kpi-subtext {
        font-size: 0.9rem;
        color: #888;
        margin-top: 0.4rem;
    }
    </style>
    """, unsafe_allow_html=True
)

col1, col2 = st.columns(2)

sp_l, col1, col2, sp_r = st.columns([1, 3, 3, 1], gap="large")
sp_l.empty()
sp_r.empty()

# Max Profit / year & 56k transactions
with col1:
    st.markdown(
        f"""
        <div class='kpi-title'>
            Max Profit / year</br>
            (EU ≈ 83B transactions / year)
        </div>
        <div class='kpi-value'>
            <span class='badge-profit' style='color:{profit_color_ann}'>
                $ {int(kpi_ann):,}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    delta_arrow = "↑" if profit_opt - profit_fixed >= 0 else "↓"
    delta = abs(profit_opt-profit_fixed)
    delta_color = "#008000" if profit_opt - profit_fixed >= 0 else "#FF0000"
    st.markdown(f"""
        <div class='kpi-title'>Max Profit / 56k transactions</br>Sample</div>
        <div class='kpi-value'>
    <span class='badge-profit' style='color:{profit_color_56k}'>
        $ {int(profit_opt):,}
    </span>
</div>
        <div class='kpi-subtext'>
          <span style="color:{delta_color}; font-weight:bold">
            {delta_arrow} $ {delta:,.0f}
            (Threshold Optimization τ = {tau_opt:.3f})
          </span><br>
          Profit: $ {int(profit_fixed):,} @ τ = 0.5
        </div>
    """, unsafe_allow_html=True)

# Profit curve & confusion matrices
st.markdown("<div style='margin-top: 1.25rem'></div>", unsafe_allow_html=True)
st.subheader(
    "Profit vs Missed Fraud Transactions Penalty Plot "
    "--- Confusion Matrices (Optimal vs Fixed Threshold)"
)
col_plot, col_cm = st.columns([3, 1])

slider_index = int(fn_penalty_pct * 100)
x_val = pen_grid[slider_index] * 100
y_best = curve_best[slider_index]
y_fixed = curve_fixed[slider_index]

with col_plot:
    fig, ax = plt.subplots(figsize=(9.5, 6.325))
    ax.plot(
        pen_grid * 100,
        curve_best,
        label="Optimal Threshold",
        lw=2,
        c="#1f77b4"
    )
    ax.plot(
        pen_grid * 100,
        curve_fixed,
        label="Fixed Threshold = 0.5",
        ls="--",
        c="#ff7f0e"
    )
    ax.plot(x_val, y_best, "o", markersize=10, color="#1f77b4", label=None)
    ax.plot(x_val, y_fixed, "o", markersize=8, color="#ff7f0e", label=None)
    ax.plot([x_val, x_val], [y_fixed, y_best], ls=":", color="gray", lw=1.5)
    ax.set_xlabel("Missed Fraud Transaction Penalty (% of fraud $)")
    ax.set_ylabel("Profit  $/ 56k transactions")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)

with col_cm:
    # Optimal threshold confusion matrix
    fig_opt, ax_opt = plt.subplots(figsize=(3.0, 3.0))
    disp_opt = ConfusionMatrixDisplay(
        cm_opt, display_labels=["Legit", "Fraud"]
    )
    disp_opt.plot(values_format="d", cmap="Blues", ax=ax_opt, colorbar=False)
    ax_opt.set_title(f"Optimal τ = {tau_opt:.2f}", fontsize=10)
    ax_opt.set_xlabel("Predicted", fontsize=9)
    ax_opt.set_ylabel("Actual", fontsize=9)
    fig_opt.tight_layout(pad=1.0)
    st.pyplot(fig_opt)

    # Fixed threshold = 0.5 confusion matrix
    fig_fixed, ax_fixed = plt.subplots(figsize=(3.0, 3.0))
    disp_fixed = ConfusionMatrixDisplay(
        cm_fixed, display_labels=["Legit", "Fraud"]
    )
    disp_fixed.plot(
        values_format="d",
        cmap="Blues",
        ax=ax_fixed,
        colorbar=False
    )
    ax_fixed.set_title("Fixed τ = 0.50", fontsize=10)
    ax_fixed.set_xlabel("Predicted", fontsize=9)
    ax_fixed.set_ylabel("Actual", fontsize=9)
    fig_fixed.tight_layout(pad=1.0)
    st.pyplot(fig_fixed)

# Local Interpretability Explanations (LIME)
st.markdown("---")
st.subheader("Explain Individual Predictions (LIME)")

# Use cahched example transactions
idx_dict = {
    "Correctly Identified Fraud (TP)": example_idx["TP"],
    "Correctly Identified Legit (TN)": example_idx["TN"],
    "Wrongly Flagged Legit (FP)": example_idx["FP"],
    "Missed Fraud (FN)": example_idx["FN"],
}

with st.expander("Run live inference & explain model reasoning",
                 expanded=True):
    show_adv = st.checkbox(
        "Show False-Positive / False-Negative examples also",
        value=False,
        help="FP/FN also saved for deeper analysis."
    )

    visible_cats = ["Correctly Identified Fraud (TP)",
                    "Correctly Identified Legit (TN)"]
    if show_adv:
        visible_cats += ["Wrongly Flagged Legit (FP)",
                         "Missed Fraud (FN)"]

    flat_opts = [(cat, idx)
                 for cat in visible_cats
                 for idx in idx_dict[cat]]
    labels = [f"{cat} → Transaction #{idx}" for cat, idx in flat_opts]

    choice = st.selectbox("Choose a transaction", options=labels, index=0)
    chosen_cat, chosen_idx = choice.split(" → Transaction #")
    row_idx = int(chosen_idx)

    if st.button("Run live inference + explain"):

        # Live prediction
        X_row = X_test.iloc[[row_idx]]
        t0 = time.time()
        proba = model.predict_proba(X_row)[0, 1]
        latency = (time.time() - t0) * 1000
        pred_cls = "Fraud" if proba >= tau_opt else "Legit"

        # LIME explanation
        explainer = LimeTabularExplainer(
            X_test.values,
            feature_names=X_test.columns.tolist(),
            class_names=["Legit", "Fraud"],
            mode="classification",
            discretize_continuous=True,
        )
        exp = explainer.explain_instance(
            X_row.values[0], model.predict_proba, num_features=8
        )

        # LIME layout
        col_fig, col_info = st.columns([3, 2], gap="medium")

        # LIME bar-chart
        with col_fig:
            fig = exp.as_pyplot_figure()
            fig.set_figwidth(7)
            fig.set_figheight(4.5)
            st.pyplot(fig, use_container_width=True)

        # LIME info CSS
        with col_info:
            st.markdown(
                f"""
                <style>
                .col-info-wrap {{
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    height:100%;
                    padding-left:4px;
                }}
                .fact-label  {{font-size:1.15rem;font-weight:700;" +
                "margin:0 0 0.15rem 0;}}
                .fact-body   {{font-size:1.05rem;margin:0.10rem 0 0.40rem 0;}}
                .how-read    {{font-size:1.00rem;line-height:1.45;
                                margin-top:0.75rem;}}
                .badge       {{padding:2px 8px;border-radius:4px;
                font-weight:700}}
                .badge-lat   {{background:#065f46;color:#b9f6ca;}}
                .badge-cls   {{background:#1e40af;color:#bfdbfe;}}
                </style>

                <div class='col-info-wrap'>

                <p class='fact-label'>Model Prediction Latency:
                    <span class='badge badge-lat'>{latency:,.3f}&nbsp;ms</span>
                </p>

                <p class='fact-label'>Predicted&nbsp;class:
                    <span class='badge badge-cls'>{pred_cls}</span>
                </p>

                <p class='fact-body'>
                    <b>Probability of Fraud&nbsp;(τ&nbsp;=&nbsp;
                    {tau_opt:.3f}):</b>
                    {proba:.4f}
                </p>
                <p class='fact-body'><b>Transaction&nbsp;ID:</b> {row_idx}</p>
                <p class='fact-body'><b>Case&nbsp;type:</b> {chosen_cat}</p>

                <p class='how-read'>
                    <em>How to read this:</em><br>
                    Bars on the left show which features pushed the model
                    toward <b>{pred_cls}</b>.
                    Longer bars = stronger influence.</br>
                    green = feature increased the fraud score</br>
                    red = feature lowered the fraud score.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        # Top LIME contributing features
        st.markdown("### Top contributing features")

        feature_box = """
        <style>
          .feat-wrap{max-height:240px;overflow-y:auto;margin-top:0.25rem;}
          .feat-row{display:flex;justify-content:space-between;
                    font-size:0.9rem;padding:2px 0;}
          .badge{padding:0 6px;border-radius:4px;font-weight:600;color:#fff;}
        </style>
        <div class='feat-wrap'>
        """
        for f, w in exp.as_list():
            colour = "#16a34a" if w > 0 else "#dc2626"
            feature_box += (
                f"<div class='feat-row'>"
                f"<span>{f}</span>"
                f"<span class='badge' style='background:{colour}'>"
                f"{w:+.6f}</span>"
                f"</div>"
            )
        feature_box += "</div>"
        st.markdown(feature_box, unsafe_allow_html=True)

# Model Card
st.markdown("---")
st.subheader("Model Details")
with st.expander("Model Card - XGBoost + SMOTE", expanded=True):
    st.markdown(
        f"""
        <style>
        /* simple dark-mode friendly table */
        .model-card-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.94rem;
        }}
        .model-card-table th,
        .model-card-table td {{
            border: 1px solid #444;
            padding: 6px 10px;
            vertical-align: top;
        }}
        .model-card-table th {{
            width: 190px;
            background: #111;
            font-weight: 700;
        }}
        </style>

        """
        "<strong>Model:</strong> Gradient-boosted decision trees "
        "(<strong>XGBoost</strong>) trained on a "
        "<strong>highly-imbalanced</strong> European credit-card dataset.<br>"
        "Minority class up-sampled with <strong>SMOTE</strong>; decision "
        "threshold&nbsp;τ tuned for business profit."
        f"""
        <table class='model-card-table'>

        <tr><th>Dataset</th><td>
        Kaggle Dataset: <em>Credit-Card Fraud Detection</em>
        (EU card-holders, Sept&nbsp;2013).
        284k&nbsp; legitimate &rarr; 492 fraud.<br>
        Original features V1-V28 are PCA components anonymized for privacy;
        “Time” and “Amount” left clear.
        </td></tr>

        <tr><th>Feature&nbsp;Engineering</th><td>
        Scaled <em>Time</em>, log-scaled <em>Amount</em>;
        kept PCA components unchanged. No PII or protected attributes.
        </td></tr>

        <tr><th>Business&nbsp;Assumptions</th><td>
        Average fraud <strong>${avg_fraud:,}</strong>; success fee
        <strong>{tp_fee_pct:.0%}</strong> of fraud value; manual-review cost
        <strong>${fp_cost:.2f}</strong>/FP; missed-fraud penalty
        <strong>{fn_penalty_pct:.0%}</strong> of fraud value.<br>
        All amounts in <strong>USD</strong>. Customer-churn cost not included.
        </td></tr>

        <tr><th>Metric</th><td>
        <code>Profit = TP x (fee % x $fraud) - FP x $cost - FN x
        (penalty % x $fraud)</code>
        </td></tr>

        <tr><th>Performance&nbsp;(test)</th><td>
        PR-AUC ≈ 0.88 &nbsp;|&nbsp; ROC-AUC ≈ 0.97.
        τ-optimisation lifts profit by
        <strong>${profit_opt - profit_fixed:,.0f}</strong>
        on the 56k hold-out sample
        vs. fixed&nbsp;τ = 0.5.
        </td></tr>

        <tr><th>Limitations&nbsp;&amp;&nbsp;Bias</th><td>
        Data from 2013 → possible concept drift.<br>
        EU-only card network - may not generalise worldwide.<br>
        Synthetic samples from SMOTE may distort the boundary.<br>
        No customer-retention or regulatory-fine costs modelled.
        </td></tr>

        <tr><th>Responsible&nbsp;AI</th><td>
        No sensitive attributes used; per-transaction explanations via LIME.
        Business owners can tune τ to balance risk and reward.
        </td></tr>

        <tr><th>Deployment</th><td>
        Model version <strong>1</strong> registered in
        MLflow (Production) as well as pickle file.
        </td></tr>

        <tr><th>Potential&nbsp;Improvements</th><td>
        Further bayesian hyper-parameter tuning (one parameter at a time)<br>
        Time-aware models (LSTM/Transformer) or Isolation-Forest pre-screen<br>
        Analyze FP/FN clusters to craft new features<br>
        Currency-aware profit tracking &amp; regional drift monitoring<br>
        Ensemble stacking &amp; cost-sensitive losses<br>
        Validate cost assumptions with stakeholders
        </td></tr>

        </table>
        """,
        unsafe_allow_html=True
    )
