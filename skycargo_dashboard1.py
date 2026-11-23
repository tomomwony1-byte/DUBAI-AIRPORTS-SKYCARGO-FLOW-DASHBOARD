import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Emirates SkyCargo | Dubai Hub Flow Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------------
# 1. SYNTHETIC EMIRATES SKYCARGO UAE HUB DATASET (ORIGIN -> HUB -> DEST)
# -------------------------------------------------------------------------
@st.cache_data
def generate_skycargo_data(n_shipments: int = 6000, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hubs = ["DXB (Dubai Intl)", "DWC (Al Maktoum)"]
    hub_probs = [0.64, 0.36]

    regions = [
        "Far East & China",
        "Europe",
        "South Asia",
        "Americas",
        "Middle East & Africa",
    ]
    origin_probs = [0.28, 0.24, 0.18, 0.15, 0.15]
    dest_probs = [0.14, 0.28, 0.15, 0.21, 0.22]

    products = [
        ("General Cargo", 8.0, 0.42),
        ("Emirates Courier / E-Commerce", 5.5, 0.26),
        ("Emirates Fresh (Perishables)", 4.5, 0.19),
        ("Emirates Pharma (Cold Chain)", 4.0, 0.13),
    ]
    prod_names = [p[0] for p in products]
    prod_slas = {p[0]: p[1] for p in products}
    prod_probs = [p[2] for p in products]

    dates = pd.Timestamp("2026-10-22") + pd.to_timedelta(rng.integers(0, 30, size=n_shipments), unit="D")
    assigned_hubs = rng.choice(hubs, size=n_shipments, p=hub_probs)
    origins = rng.choice(regions, size=n_shipments, p=origin_probs)
    destinations = rng.choice(regions, size=n_shipments, p=dest_probs)
    assigned_products = rng.choice(prod_names, size=n_shipments, p=prod_probs)

    directions = rng.choice(["Inbound", "Outbound"], size=n_shipments, p=[0.51, 0.49])

    base_tonnes = np.where(
        assigned_hubs == "DWC (Al Maktoum)",
        rng.gamma(shape=2.6, scale=1.9, size=n_shipments),
        rng.gamma(shape=2.2, scale=1.4, size=n_shipments),
    )
    tonnes = np.clip(base_tonnes, 0.3, 28.0).round(2)

    sla_targets = np.array([prod_slas[p] for p in assigned_products])
    dwell_hours = sla_targets * rng.uniform(0.55, 1.15, size=n_shipments)
    congestion_mask = rng.random(n_shipments) < 0.09
    dwell_hours[congestion_mask] += rng.uniform(2.0, 5.5, size=congestion_mask.sum())
    dwell_hours = dwell_hours.round(2)

    return pd.DataFrame({
        "awb_batch_id": [f"176-{400000 + i}" for i in range(n_shipments)],
        "date": dates,
        "hub": assigned_hubs,
        "direction": directions,
        "origin_region": origins,
        "dest_region": destinations,
        "product": assigned_products,
        "tonnes": tonnes,
        "sla_dwell_hrs": sla_targets,
        "dwell_hours": dwell_hours,
    })


def generate_executive_anomalies(data: pd.DataFrame) -> list[tuple[str, str]]:
    """Detects operational anomalies in the filtered SkyCargo slice."""
    alerts = []

    # 1. Check Inbound vs. Outbound Hub Balance
    in_t = data.loc[data["direction"] == "Inbound", "tonnes"].sum()
    out_t = data.loc[data["direction"] == "Outbound", "tonnes"].sum()
    ratio = (out_t / in_t * 100) if in_t > 0 else 100.0
    net_t = in_t - out_t

    if ratio < 92.0:
        alerts.append((
            "warning",
            f"**Warehouse Accumulation Alert:** Outbound uplift is trailing inbound arrivals "
            f"(**{ratio:.1f}%** throughput ratio, **+{net_t:,.0f} T** net buildup). Recommend allocating "
            f"additional widebody bellyhold or DWC charter capacity."
        ))
    elif ratio > 103.0:
        alerts.append((
            "info",
            f"**Backlog Drawdown:** Outbound uplift exceeds inbound arrivals (**{ratio:.1f}%** ratio, "
            f"**{net_t:,.0f} T** net clearance), actively clearing prior terminal inventory."
        ))
    else:
        alerts.append((
            "success",
            f"**Balanced Hub Throughput:** Outbound/Inbound flow ratio is healthy at **{ratio:.1f}%** "
            f"(net variance **{net_t:+,.0f} T**), keeping DXB/DWC ramp buffers within target."
        ))

    # 2. Identify Worst-Performing Hub x Product SLA Bottleneck
    seg = (
        data.groupby(["hub", "product"], as_index=False)
        .agg(
            avg_dwell=("sim_dwell_hours", "mean"),
            sla_target=("sla_dwell_hrs", "first"),
            sla_pct=("sla_met", lambda s: s.mean() * 100),
            breaches=("sla_met", lambda s: (~s).sum()),
        )
        .sort_values("sla_pct", ascending=True)
    )
    worst = seg.iloc[0]
    if worst["sla_pct"] < 88.0:
        alerts.append((
            "warning",
            f"**SLA Bottleneck Detected — {worst['hub']} ({worst['product']}):** Compliance dropped to "
            f"**{worst['sla_pct']:.1f}%** ({int(worst['breaches'])} breached batches; avg dwell "
            f"**{worst['avg_dwell']:.2f}h** vs **{worst['sla_target']:.1f}h** SLA). Prioritize customs pre-clearance."
        ))

    # 3. Spot Peak Daily Congestion Spike
    daily_dwell = data.groupby("date", as_index=False).agg(
        avg_dwell=("sim_dwell_hours", "mean"),
        vol=("tonnes", "sum"),
    )
    peak_day = daily_dwell.loc[daily_dwell["avg_dwell"].idxmax()]
    overall_avg = data["sim_dwell_hours"].mean()
    if peak_day["avg_dwell"] > overall_avg * 1.12:
        alerts.append((
            "warning",
            f"**Peak Dwell Spike on {peak_day['date'].strftime('%b %d, %Y')}:** Average dwell jumped to "
            f"**{peak_day['avg_dwell']:.2f} hrs** (+{((peak_day['avg_dwell']/overall_avg)-1)*100:.0f}% above period average) "
            f"across **{peak_day['vol']:,.0f} T** handled."
        ))

    return alerts


df = generate_skycargo_data()

# -------------------------------------------------------------------------
# 2. SIDEBAR FILTERS, SIMULATOR & CSV EXPORT
# -------------------------------------------------------------------------
st.sidebar.header("UAE Hub Filters")

selected_hubs = st.sidebar.multiselect(
    "Dubai Hub Terminal",
    options=sorted(df["hub"].unique()),
    default=sorted(df["hub"].unique()),
)

selected_products = st.sidebar.multiselect(
    "SkyCargo Product Vertical",
    options=sorted(df["product"].unique()),
    default=sorted(df["product"].unique()),
)

selected_origins = st.sidebar.multiselect(
    "Origin Region",
    options=sorted(df["origin_region"].unique()),
    default=sorted(df["origin_region"].unique()),
)

st.sidebar.divider()
st.sidebar.header("What-If Hub Flow Simulator")
dwell_improvement_pct = st.sidebar.slider(
    "Customs & Ramp Transfer Speedup (%)",
    min_value=0,
    max_value=30,
    value=0,
    step=5,
)

filtered_df = df[
    (df["hub"].isin(selected_hubs))
    & (df["product"].isin(selected_products))
    & (df["origin_region"].isin(selected_origins))
].copy()

if filtered_df.empty:
    st.warning("No cargo records match the selected filters. Reset filters in the sidebar.")
    st.stop()

filtered_df["sim_dwell_hours"] = (filtered_df["dwell_hours"] * (1 - dwell_improvement_pct / 100.0)).round(2)
filtered_df["sla_met"] = filtered_df["sim_dwell_hours"] <= filtered_df["sla_dwell_hrs"]

# Sidebar CSV Export Button
st.sidebar.divider()
st.sidebar.header("Export Filtered Data")
csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download Filtered Cargo CSV",
    data=csv_bytes,
    file_name="emirates_skycargo_filtered_flow.csv",
    mime="text/csv",
    use_container_width=True,
)

# -------------------------------------------------------------------------
# 3. HEADER & TOP KPI CARDS
# -------------------------------------------------------------------------
st.title("Emirates SkyCargo — UAE Cargo In & Out Flow Monitor")
st.caption(
    f"Dubai Dual-Hub Operations (**DXB & DWC**) | Analyzing **{len(filtered_df):,}** ULD/AWB batches "
    f"| Clearance Speedup Simulation: **-{dwell_improvement_pct}% dwell time**"
)

inbound_tonnes = filtered_df.loc[filtered_df["direction"] == "Inbound", "tonnes"].sum()
outbound_tonnes = filtered_df.loc[filtered_df["direction"] == "Outbound", "tonnes"].sum()
net_balance = inbound_tonnes - outbound_tonnes
throughput_ratio = (outbound_tonnes / inbound_tonnes * 100) if inbound_tonnes > 0 else 0

avg_dwell = filtered_df["sim_dwell_hours"].mean()
baseline_dwell = filtered_df["dwell_hours"].mean()
sla_compliance = filtered_df["sla_met"].mean() * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Inbound Cargo (UAE)", f"{inbound_tonnes:,.0f} T", "Arrivals & Imports")
k2.metric("Total Outbound Cargo (UAE)", f"{outbound_tonnes:,.0f} T", f"{net_balance:+,.0f} T Net Hub Balance")
k3.metric("Outbound / Inbound Ratio", f"{throughput_ratio:.1f}%", "Target: 92%–102% Hub Balance")
k4.metric(
    "Avg Hub Dwell & SLA Rate",
    f"{avg_dwell:.2f} hrs",
    f"{sla_compliance:.1f}% On-Time SLA" + (f" (-{baseline_dwell - avg_dwell:.2f}h)" if dwell_improvement_pct > 0 else ""),
)

# -------------------------------------------------------------------------
# 4. AUTOMATED EXECUTIVE ANOMALY SUMMARY PANEL
# -------------------------------------------------------------------------
with st.expander("Automated Executive Anomaly & SLA Brief", expanded=True):
    anomalies = generate_executive_anomalies(filtered_df)
    for level, message in anomalies:
        if level == "warning":
            st.warning(message)
        elif level == "success":
            st.success(message)
        else:
            st.info(message)

st.divider()

# -------------------------------------------------------------------------
# 5. SANKEY DIAGRAM: ORIGIN REGIONS -> DXB/DWC HUBS -> DESTINATION REGIONS
# -------------------------------------------------------------------------
st.subheader("Global Air-Freight Transshipment Flow: Origin → Dubai Hubs → Destination (Tonnes)")

origins_list = sorted(filtered_df["origin_region"].unique())
hubs_list = sorted(filtered_df["hub"].unique())
dests_list = sorted(filtered_df["dest_region"].unique())

node_labels = (
    [f"{r} (Origin)" for r in origins_list]
    + hubs_list
    + [f"{r} (Dest)" for r in dests_list]
)
node_idx = {label: i for i, label in enumerate(node_labels)}

node_colors = (
    ["#0284c7"] * len(origins_list)
    + ["#d71920"] * len(hubs_list)
    + ["#059669"] * len(dests_list)
)

leg1 = filtered_df.groupby(["origin_region", "hub"], as_index=False)["tonnes"].sum()
leg2 = filtered_df.groupby(["hub", "dest_region"], as_index=False)["tonnes"].sum()

sources, targets, values, link_colors = [], [], [], []

for _, row in leg1.iterrows():
    sources.append(node_idx[f"{row['origin_region']} (Origin)"])
    targets.append(node_idx[row["hub"]])
    values.append(round(row["tonnes"], 1))
    link_colors.append("rgba(2, 132, 199, 0.28)")

for _, row in leg2.iterrows():
    sources.append(node_idx[row["hub"]])
    targets.append(node_idx[f"{row['dest_region']} (Dest)"])
    values.append(round(row["tonnes"], 1))
    link_colors.append("rgba(215, 25, 32, 0.25)")

fig_sankey = go.Figure(
    data=[
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=22,
                line=dict(color="white", width=0.8),
                label=node_labels,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            ),
        )
    ]
)
fig_sankey.update_layout(height=400, margin=dict(l=10, r=10, t=15, b=15))
st.plotly_chart(fig_sankey, use_container_width=True)

st.divider()

# -------------------------------------------------------------------------
# 6. DAILY IN/OUT FLOW TREND & PRODUCT VERTICAL COMPARISON
# -------------------------------------------------------------------------
mid_left, mid_right = st.columns([1.35, 1])

with mid_left:
    st.subheader("Daily Inbound vs. Outbound Tonnage Flow")
    daily_flow = (
        filtered_df.groupby(["date", "direction"], as_index=False)["tonnes"]
        .sum()
        .pivot(index="date", columns="direction", values="tonnes")
        .fillna(0)
        .reset_index()
        .sort_values("date")
    )
    if "Inbound" not in daily_flow:
        daily_flow["Inbound"] = 0.0
    if "Outbound" not in daily_flow:
        daily_flow["Outbound"] = 0.0
    daily_flow["Net_Flow"] = daily_flow["Inbound"] - daily_flow["Outbound"]

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(x=daily_flow["date"], y=daily_flow["Inbound"], name="Inbound (T)", marker_color="#0ea5e9"))
    fig_daily.add_trace(go.Bar(x=daily_flow["date"], y=daily_flow["Outbound"], name="Outbound (T)", marker_color="#d71920"))
    fig_daily.add_trace(
        go.Scatter(
            x=daily_flow["date"],
            y=daily_flow["Net_Flow"],
            name="Net Hub Accumulation",
            mode="lines+markers",
            line=dict(color="#10b981", width=2),
            yaxis="y2",
        )
    )
    fig_daily.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Date",
        yaxis=dict(title="Cargo Volume (Tonnes)"),
        yaxis2=dict(title="Net Balance (Tonnes)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_daily, use_container_width=True)

with mid_right:
    st.subheader("In & Out Flow by Product Vertical")
    prod_flow = filtered_df.groupby(["product", "direction"], as_index=False)["tonnes"].sum()
    fig_prod = px.bar(
        prod_flow,
        x="tonnes",
        y="product",
        color="direction",
        barmode="group",
        orientation="h",
        color_discrete_map={"Inbound": "#0ea5e9", "Outbound": "#d71920"},
        text_auto=".0f",
    )
    fig_prod.update_layout(
        height=360,
        margin=dict(l=10, r=20, t=20, b=10),
        xaxis_title="Total Tonnage (T)",
        yaxis_title="",
        legend_title="Flow",
    )
    st.plotly_chart(fig_prod, use_container_width=True)

# -------------------------------------------------------------------------
# 7. BOTTOM ROW: TERMINAL SLA SCORECARD & DATA PREVIEW
# -------------------------------------------------------------------------
st.subheader("DXB / DWC Terminal & Cold-Chain SLA Scorecard")
scorecard = (
    filtered_df.groupby(["hub", "product"], as_index=False)
    .agg(
        inbound_t=("tonnes", lambda s: s[filtered_df.loc[s.index, "direction"] == "Inbound"].sum()),
        outbound_t=("tonnes", lambda s: s[filtered_df.loc[s.index, "direction"] == "Outbound"].sum()),
        avg_dwell_hrs=("sim_dwell_hours", "mean"),
        sla_pct=("sla_met", lambda s: s.mean() * 100),
    )
    .sort_values("sla_pct", ascending=True)
)

st.dataframe(
    scorecard.style.format({
        "inbound_t": "{:,.0f} T",
        "outbound_t": "{:,.0f} T",
        "avg_dwell_hrs": "{:.2f} h",
        "sla_pct": "{:.1f}%",
    }).background_gradient(subset=["sla_pct"], cmap="RdYlGn", vmin=75, vmax=98),
    use_container_width=True,
    hide_index=True,
)