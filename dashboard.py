"""dashboard.py — Bonus: a small analytics dashboard built from logs/tickets.csv."""
import altair as alt
import pandas as pd
import streamlit as st

# Status colours: green = Resolved, red = Escalated (always shown with a legend label)
STATUS_SCALE = alt.Scale(domain=["Resolved", "Escalated"], range=["#0ca30c", "#d03b3b"])
BAR_COLOR = "#2a78d6"  # one colour for single-series charts
URGENCY_ORDER = ["Low", "Medium", "High"]
URGENCY_SCORE = {"Low": 1, "Medium": 2, "High": 3}

ROW = alt.Step(34)                        # fixed height per bar row
BAND = alt.Scale(paddingInner=0.45)       # gap between bars -> thin bars


def tickets_per_category(log: pd.DataFrame) -> alt.Chart:
    data = log.groupby(["category", "status"]).size().reset_index(name="tickets")
    order = log["category"].value_counts().index.tolist()  # biggest category first
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("tickets:Q", title="Tickets", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("category:N", sort=order, title=None, scale=BAND),
            color=alt.Color("status:N", scale=STATUS_SCALE,
                            legend=alt.Legend(title=None, orient="top")),
            order=alt.Order("status:N", sort="descending"),  # Resolved first, then Escalated
            tooltip=[alt.Tooltip("category:N", title="Category"),
                     alt.Tooltip("status:N", title="Status"),
                     alt.Tooltip("tickets:Q", title="Tickets")],
        )
        .properties(height=ROW)
    )


def urgency_distribution(log: pd.DataFrame) -> alt.Chart:
    counts = (log["urgency"].value_counts()
              .reindex(URGENCY_ORDER, fill_value=0)
              .rename_axis("urgency").reset_index(name="tickets"))
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=4, color=BAR_COLOR)
        .encode(
            x=alt.X("tickets:Q", title="Tickets", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("urgency:N", sort=URGENCY_ORDER, title=None, scale=BAND),
            tooltip=[alt.Tooltip("urgency:N", title="Urgency"),
                     alt.Tooltip("tickets:Q", title="Tickets")],
        )
        .properties(height=ROW)
    )


def escalation_reasons(log: pd.DataFrame) -> alt.Chart | None:
    reasons = (log.loc[log["status"] == "Escalated", "escalation_reason"]
               .str.split("+").explode().str.strip())
    reasons = reasons[reasons != ""]
    if reasons.empty:
        return None
    counts = reasons.value_counts().rename_axis("reason").reset_index(name="tickets")
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=4, color=BAR_COLOR)
        .encode(
            x=alt.X("tickets:Q", title="Escalated tickets with this reason",
                    axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("reason:N", sort="-x", title=None, scale=BAND,
                    axis=alt.Axis(labelLimit=220)),
            tooltip=[alt.Tooltip("reason:N", title="Reason"),
                     alt.Tooltip("tickets:Q", title="Tickets")],
        )
        .properties(height=ROW)
    )


def render_dashboard(log: pd.DataFrame) -> None:
    if log.empty:
        st.info("No tickets yet. Submit one from the New Ticket tab.")
        return

    total = len(log)
    escalated = int((log["status"] == "Escalated").sum())
    resolved = total - escalated
    avg_urgency = log["urgency"].map(URGENCY_SCORE).dropna().mean()
    avg_label = URGENCY_ORDER[int(round(avg_urgency)) - 1] if pd.notna(avg_urgency) else "—"

    # ---- headline numbers ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total tickets", total, border=True)
    k2.metric("Resolved", f"{resolved / total:.0%}", f"{resolved} tickets",
              delta_color="off", delta_arrow="off", border=True)
    k3.metric("Escalated", f"{escalated / total:.0%}", f"{escalated} tickets",
              delta_color="off", delta_arrow="off", border=True)
    k4.metric("Average urgency", f"{avg_urgency:.1f} / 3", avg_label,
              delta_color="off", delta_arrow="off", border=True,
              help="Low = 1, Medium = 2, High = 3")

    # ---- charts ----
    left, right = st.columns(2)
    with left:
        st.markdown("**Tickets per category**")
        st.altair_chart(tickets_per_category(log), width="stretch")
    with right:
        st.markdown("**Urgency**")
        st.altair_chart(urgency_distribution(log), width="stretch")

    st.markdown("**Why tickets were escalated**")
    chart = escalation_reasons(log)
    if chart is None:
        st.caption("No escalated tickets yet.")
    else:
        st.altair_chart(chart, width="stretch")
        st.caption("One ticket can have several reasons, so bars add up to more than the escalated total.")