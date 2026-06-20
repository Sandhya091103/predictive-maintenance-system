import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d2e 0%, #12151f 100%);
    border-right: 1px solid #2d3147;
}

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
    border: 1px solid #2d3147;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin: 4px 0;
}
.metric-card .label {
    font-size: 12px;
    color: #8b93c4;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 28px;
    font-weight: 700;
    margin: 0;
}

/* Alert cards */
.alert-card {
    border-radius: 14px;
    padding: 22px 26px;
    margin: 10px 0;
}
.alert-card .alert-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 6px;
}
.alert-card .alert-sub {
    font-size: 14px;
    opacity: 0.9;
}

/* Sensor bar */
.sensor-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 8px 0;
    font-size: 13px;
}

/* Section headers */
.section-header {
    font-size: 16px;
    font-weight: 700;
    color: #c5cae9;
    border-bottom: 2px solid #2d3147;
    padding-bottom: 8px;
    margin: 20px 0 12px 0;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* App header */
.app-header {
    background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
    border-bottom: 1px solid #2d3147;
    padding: 18px 0 14px 0;
    margin-bottom: 24px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1d2e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b93c4;
    border-radius: 8px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #2d3580 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH       = "model/maintenance_model.pkl"
HIGH_THRESHOLD   = 0.70
MEDIUM_THRESHOLD = 0.35
MAX_TOOL_WEAR    = 253
TYPE_MAP         = {"H": 0, "L": 1, "M": 2}

FEATURES = [
    "Air_temperature_K", "Process_temperature_K", "Rotational_speed_rpm",
    "Torque_Nm", "Tool_wear_min", "Type_encoded", "temp_delta", "power",
    "wear_rate", "RUL", "torque_wear", "rpm_torque", "temp_ratio",
    "rolling_temp_5", "rolling_torque_10", "lag_torque_1", "lag_rpm_1",
]

# Normal ranges from dataset (approx 25th–75th percentile)
NORMAL_RANGES = {
    "Air_temperature_K":    (298.3, 301.5),
    "Process_temperature_K":(308.8, 311.1),
    "Rotational_speed_rpm": (1423,  1612),
    "Torque_Nm":            (33.2,  46.8),
    "Tool_wear_min":        (53,    162),
}

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def build_features(air_temp, proc_temp, rpm, torque, tool_wear, machine_type):
    te = TYPE_MAP[machine_type]
    return pd.DataFrame([{
        "Air_temperature_K":    air_temp,
        "Process_temperature_K": proc_temp,
        "Rotational_speed_rpm": rpm,
        "Torque_Nm":            torque,
        "Tool_wear_min":        tool_wear,
        "Type_encoded":         te,
        "temp_delta":           proc_temp - air_temp,
        "power":                torque * (2 * np.pi * rpm / 60),
        "wear_rate":            tool_wear / (rpm + 1),
        "RUL":                  MAX_TOOL_WEAR - tool_wear,
        "torque_wear":          torque * tool_wear,
        "rpm_torque":           rpm * torque,
        "temp_ratio":           proc_temp / air_temp,
        "rolling_temp_5":       air_temp,
        "rolling_torque_10":    torque,
        "lag_torque_1":         torque,
        "lag_rpm_1":            rpm,
    }])[FEATURES]

def gauge_chart(prob):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("#12151f")
    ax.set_facecolor("#12151f")

    # Background arc
    for i, (lo, hi, c) in enumerate([(0,.35,"#1a3a2a"),(0.35,.70,"#3a2e10"),(0.70,1.0,"#3a1212")]):
        t = np.linspace(np.pi*(1-hi), np.pi*(1-lo), 80)
        ax.plot(np.cos(t), np.sin(t), color=c, linewidth=28, solid_capstyle="butt", zorder=1)

    # Foreground arc
    for lo, hi, c in [(0,.35,"#2ecc71"),(0.35,.70,"#f39c12"),(0.70,1.0,"#e74c3c")]:
        if prob > lo:
            hi2 = min(prob, hi)
            t = np.linspace(np.pi*(1-hi2), np.pi*(1-lo), 80)
            ax.plot(np.cos(t), np.sin(t), color=c, linewidth=18, solid_capstyle="butt", zorder=2, alpha=0.95)

    # Needle
    angle = np.pi * (1 - prob)
    ax.annotate("", xy=(0.62*np.cos(angle), 0.62*np.sin(angle)), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, mutation_scale=18), zorder=5)
    ax.plot(0, 0, "o", color="white", markersize=8, zorder=6)

    # Labels
    for val, lbl in [(0,"0"),(0.5,"50"),(1,"100")]:
        a = np.pi*(1-val)
        ax.text(0.88*np.cos(a), 0.88*np.sin(a), lbl+"%",
                ha="center", va="center", fontsize=7, color="#8b93c4")

    color = "#e74c3c" if prob>=HIGH_THRESHOLD else "#f39c12" if prob>=MEDIUM_THRESHOLD else "#2ecc71"
    ax.text(0, -0.18, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=24, fontweight="bold", color=color)
    ax.text(0, -0.42, "Failure Probability", ha="center", va="center",
            fontsize=9, color="#8b93c4")

    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.6, 1.1); ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig

def sensor_health_chart(air_temp, proc_temp, rpm, torque, tool_wear):
    sensors = {
        "Air Temp (K)":       (air_temp,  *NORMAL_RANGES["Air_temperature_K"],    295.3, 304.5),
        "Process Temp (K)":   (proc_temp, *NORMAL_RANGES["Process_temperature_K"], 305.7, 313.8),
        "RPM":                (rpm,       *NORMAL_RANGES["Rotational_speed_rpm"],  1168,  2886),
        "Torque (Nm)":        (torque,    *NORMAL_RANGES["Torque_Nm"],             3.8,   76.6),
        "Tool Wear (min)":    (tool_wear, *NORMAL_RANGES["Tool_wear_min"],         0,     253),
    }

    fig, axes = plt.subplots(len(sensors), 1, figsize=(6, 4.5))
    fig.patch.set_facecolor("#12151f")

    for ax, (name, (val, lo, hi, vmin, vmax)) in zip(axes, sensors.items()):
        ax.set_facecolor("#12151f")
        norm_val = (val - vmin) / (vmax - vmin + 1e-9)
        norm_lo  = (lo  - vmin) / (vmax - vmin + 1e-9)
        norm_hi  = (hi  - vmin) / (vmax - vmin + 1e-9)

        # Track
        ax.barh(0, 1, height=0.5, color="#1e2235", left=0, zorder=1)
        # Normal zone
        ax.barh(0, norm_hi-norm_lo, height=0.5, left=norm_lo, color="#1a3a2a", zorder=2)
        # Value bar
        col = "#2ecc71" if norm_lo <= norm_val <= norm_hi else "#e74c3c"
        ax.barh(0, norm_val, height=0.4, color=col, zorder=3, alpha=0.85)

        ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([]); ax.set_xticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(-0.02, 0, name, ha="right", va="center", fontsize=8, color="#c5cae9",
                transform=ax.get_yaxis_transform())
        ax.text(1.02, 0, f"{val:.0f}", ha="left", va="center", fontsize=8, color=col,
                fontweight="bold", transform=ax.get_yaxis_transform())

    fig.text(0.5, -0.02, "◼ Normal range    ◼ Current value",
             ha="center", fontsize=7, color="#8b93c4")
    plt.tight_layout(pad=0.5)
    return fig

def importance_chart(model):
    imp  = model.feature_importances_
    feat = pd.Series(imp, index=FEATURES).sort_values(ascending=True).tail(10)
    eng  = {"torque_wear","power","RUL","wear_rate","temp_delta","rpm_torque","temp_ratio"}

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#12151f")
    ax.set_facecolor("#12151f")

    colors = ["#e74c3c" if f in eng else "#3d5afe" for f in feat.index]
    bars = ax.barh(feat.index, feat.values, color=colors, edgecolor="none", height=0.6)

    ax.set_title("Feature Importances", color="white", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="#c5cae9", labelsize=8)
    ax.xaxis.label.set_color("#8b93c4")
    ax.set_xlabel("Score", fontsize=8, color="#8b93c4")
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis="x", colors="#2d3147")

    p1 = mpatches.Patch(color="#e74c3c", label="Engineered")
    p2 = mpatches.Patch(color="#3d5afe", label="Raw sensor")
    ax.legend(handles=[p1,p2], fontsize=7, framealpha=0,
              labelcolor="white", loc="lower right")

    plt.tight_layout()
    return fig

# ── Load model ─────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model not found — run the notebook first to train and save the model.")
    st.stop()

model = load_model()

# ── App header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:24px;">
    <div style="font-size:42px;">⚙️</div>
    <div>
        <div style="font-size:26px; font-weight:800; color:white; line-height:1.1;">
            Predictive Maintenance System
        </div>
        <div style="font-size:13px; color:#8b93c4; margin-top:3px;">
            AI4I 2020 Dataset &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; 98.9% Accuracy &nbsp;·&nbsp; Real-time Failure Prediction
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">Sensor Inputs</div>', unsafe_allow_html=True)

    machine_type = st.selectbox("Machine Type", ["L", "M", "H"],
                                help="L = Low, M = Medium, H = High quality variant")

    st.markdown("**Temperature**")
    air_temp  = st.slider("Air Temperature (K)",  295.0, 305.0, 300.0, 0.1)
    proc_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)

    st.markdown("**Mechanical**")
    rpm       = st.slider("Rotational Speed (RPM)", 1168, 2886, 1538)
    torque    = st.slider("Torque (Nm)", 3.8, 76.6, 40.0, 0.1)
    tool_wear = st.slider("Tool Wear (min)", 0, 253, 108)

    st.divider()
    st.markdown("**Alert Threshold**")
    threshold = st.slider("Decision threshold", 0.10, 0.90, MEDIUM_THRESHOLD, 0.01,
                          help="Lower = more sensitive (catches more failures, more false alarms)")

    st.divider()
    rul   = MAX_TOOL_WEAR - tool_wear
    power = torque * (2 * np.pi * rpm / 60)
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Remaining Useful Life</div>
        <div class="value" style="color:{'#2ecc71' if rul>100 else '#f39c12' if rul>50 else '#e74c3c'}">
            {rul} min
        </div>
    </div>
    <div class="metric-card" style="margin-top:8px;">
        <div class="label">Power Output</div>
        <div class="value" style="color:#3d5afe;">{power:.0f} W</div>
    </div>
    """, unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
X_input = build_features(air_temp, proc_temp, rpm, torque, tool_wear, machine_type)
prob    = float(model.predict_proba(X_input)[0][1])

color  = "#e74c3c" if prob>=HIGH_THRESHOLD else "#f39c12" if prob>=MEDIUM_THRESHOLD else "#2ecc71"
level  = "HIGH RISK"   if prob>=HIGH_THRESHOLD else "MEDIUM RISK" if prob>=MEDIUM_THRESHOLD else "LOW RISK"
icon   = "🔴" if prob>=HIGH_THRESHOLD else "🟡" if prob>=MEDIUM_THRESHOLD else "🟢"
action = ("Immediate maintenance required — stop machine if possible"
          if prob>=HIGH_THRESHOLD else
          "Schedule maintenance within 7 days"
          if prob>=MEDIUM_THRESHOLD else
          "Machine operating normally — continue monitoring")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍  Live Prediction", "📂  Batch Analysis"])

# ══════════════════════════════════════════════
#  TAB 1
# ══════════════════════════════════════════════
with tab1:

    # Alert banner
    st.markdown(f"""
    <div class="alert-card" style="background:{color}18; border:1.5px solid {color}60;">
        <div class="alert-title" style="color:{color};">{icon} {level}</div>
        <div class="alert-sub" style="color:#cccccc;">{action}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.3, 1.4, 1.3])

    with col1:
        st.markdown('<div class="section-header">Failure Probability</div>', unsafe_allow_html=True)
        st.pyplot(gauge_chart(prob), use_container_width=True)

        # Mini metric row
        m1, m2 = st.columns(2)
        m1.markdown(f"""
        <div class="metric-card">
            <div class="label">Prediction</div>
            <div class="value" style="color:{color}; font-size:18px;">
                {'FAILURE' if prob >= threshold else 'NORMAL'}
            </div>
        </div>""", unsafe_allow_html=True)
        m2.markdown(f"""
        <div class="metric-card">
            <div class="label">Threshold</div>
            <div class="value" style="color:#8b93c4; font-size:18px;">{threshold:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">Sensor Health</div>', unsafe_allow_html=True)
        st.pyplot(sensor_health_chart(air_temp, proc_temp, rpm, torque, tool_wear),
                  use_container_width=True)

        st.markdown("""
        <div style="font-size:11px; color:#8b93c4; margin-top:4px;">
            Green bar = within normal range &nbsp;|&nbsp; Red bar = outside normal range
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        st.pyplot(importance_chart(model), use_container_width=True)

    st.divider()

    # Derived metrics row
    st.markdown('<div class="section-header">Computed Features</div>', unsafe_allow_html=True)
    d1, d2, d3, d4, d5 = st.columns(5)
    temp_delta  = proc_temp - air_temp
    wear_rate   = tool_wear / (rpm + 1)
    torque_wear = torque * tool_wear

    for col, lbl, val, clr in [
        (d1, "Power (W)",         f"{power:.0f}",        "#3d5afe"),
        (d2, "RUL (min)",         f"{rul}",              "#2ecc71" if rul>100 else "#f39c12" if rul>50 else "#e74c3c"),
        (d3, "Temp Delta (K)",    f"{temp_delta:.1f}",   "#f39c12"),
        (d4, "Wear Rate",         f"{wear_rate:.4f}",    "#9c27b0"),
        (d5, "Torque × Wear",     f"{torque_wear:.0f}",  "#e74c3c" if torque_wear>5000 else "#2ecc71"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{lbl}</div>
            <div class="value" style="color:{clr}; font-size:22px;">{val}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — Batch
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a CSV with columns: `Type`, `Air_temperature_K`, `Process_temperature_K`,
    `Rotational_speed_rpm`, `Torque_Nm`, `Tool_wear_min`
    """)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df_b = pd.read_csv(uploaded)
        df_b.columns = df_b.columns.str.strip()

        required = {"Type","Air_temperature_K","Process_temperature_K",
                    "Rotational_speed_rpm","Torque_Nm","Tool_wear_min"}
        missing = required - set(df_b.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            rows = [build_features(r["Air_temperature_K"], r["Process_temperature_K"],
                                   r["Rotational_speed_rpm"], r["Torque_Nm"],
                                   r["Tool_wear_min"], r.get("Type","M"))
                    for _, r in df_b.iterrows()]
            X_b    = pd.concat(rows, ignore_index=True)
            probs_b = model.predict_proba(X_b)[:, 1]

            df_b["Failure_Probability"] = probs_b.round(4)
            df_b["Priority"] = pd.cut(probs_b, bins=[-1, MEDIUM_THRESHOLD, HIGH_THRESHOLD, 2],
                                      labels=["LOW","MEDIUM","HIGH"])
            df_b["Maintenance_Days"] = df_b["Priority"].map({"HIGH":1,"MEDIUM":7,"LOW":30})
            df_b["Prediction"] = (probs_b >= threshold).astype(int)

            n_high = (df_b["Priority"]=="HIGH").sum()
            n_med  = (df_b["Priority"]=="MEDIUM").sum()
            n_low  = (df_b["Priority"]=="LOW").sum()

            st.markdown("")
            c1, c2, c3, c4 = st.columns(4)
            for col, lbl, val, clr in [
                (c1, "Total Machines", len(df_b),  "#3d5afe"),
                (c2, "🔴 HIGH Risk",   n_high,     "#e74c3c"),
                (c3, "🟡 MEDIUM Risk", n_med,      "#f39c12"),
                (c4, "🟢 LOW Risk",    n_low,      "#2ecc71"),
            ]:
                col.markdown(f"""
                <div class="metric-card">
                    <div class="label">{lbl}</div>
                    <div class="value" style="color:{clr};">{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            bc1, bc2 = st.columns([1.8, 1.2])
            with bc1:
                st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
                st.dataframe(
                    df_b.sort_values("Failure_Probability", ascending=False)
                        .style.background_gradient(subset=["Failure_Probability"],
                                                   cmap="RdYlGn_r"),
                    use_container_width=True, height=360
                )
                csv_out = df_b.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results", csv_out,
                                   "maintenance_predictions.csv", "text/csv",
                                   use_container_width=True)

            with bc2:
                st.markdown('<div class="section-header">Distribution</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                fig.patch.set_facecolor("#12151f")
                ax.set_facecolor("#12151f")
                n, bins, patches = ax.hist(probs_b, bins=25, edgecolor="none", alpha=0.85)
                for patch, left in zip(patches, bins):
                    patch.set_facecolor(
                        "#e74c3c" if left>=HIGH_THRESHOLD else
                        "#f39c12" if left>=MEDIUM_THRESHOLD else "#2ecc71"
                    )
                ax.axvline(HIGH_THRESHOLD,   color="#e74c3c", linestyle="--", lw=1.5, label="High 0.70")
                ax.axvline(MEDIUM_THRESHOLD, color="#f39c12", linestyle="--", lw=1.5, label="Med 0.35")
                ax.set_title("Failure Probability", color="white", fontsize=10, fontweight="bold")
                ax.tick_params(colors="#8b93c4", labelsize=8)
                for s in ax.spines.values(): s.set_visible(False)
                ax.legend(fontsize=7, framealpha=0, labelcolor="white")
                st.pyplot(fig, use_container_width=True)

                # Priority pie
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                fig2.patch.set_facecolor("#12151f")
                vals   = [n_high, n_med, n_low]
                labels = ["HIGH","MEDIUM","LOW"]
                colors = ["#e74c3c","#f39c12","#2ecc71"]
                non_zero = [(v,l,c) for v,l,c in zip(vals,labels,colors) if v>0]
                if non_zero:
                    v,l,c = zip(*non_zero)
                    ax2.pie(v, labels=l, colors=c, autopct="%1.0f%%",
                            textprops={"color":"white","fontsize":9},
                            wedgeprops={"edgecolor":"#12151f","linewidth":2})
                ax2.set_title("Priority Split", color="white", fontsize=10, fontweight="bold")
                st.pyplot(fig2, use_container_width=True)
    else:
        st.markdown("""
        <div style="border:2px dashed #2d3147; border-radius:14px;
                    padding:50px; text-align:center; color:#8b93c4;">
            <div style="font-size:40px;">📂</div>
            <div style="font-size:16px; margin-top:10px;">Drop your CSV file here</div>
            <div style="font-size:12px; margin-top:6px;">
                Supports bulk prediction for hundreds of machines at once
            </div>
        </div>
        """, unsafe_allow_html=True)
