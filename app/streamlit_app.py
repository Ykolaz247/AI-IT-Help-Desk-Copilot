import streamlit as st
import sys
from pathlib import Path

# Allow the app to import our Copilot engine
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

sys.path.append(str(SRC_DIR))

from copilot_engine import analyze_ticket


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="AI IT Help Desk Copilot",
    page_icon="🛠️",
    layout="wide"
)


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🛠️ AI IT Help Desk Copilot")

st.write(
    "AI-assisted ticket triage for support teams. "
    "Enter a support request to receive routing and "
    "priority recommendations with confidence-based "
    "human review."
)

st.divider()


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.header("System Information")

    st.write("**Routing Model:**")
    st.write("TF-IDF + Logistic Regression")

    st.write("**Priority Model:**")
    st.write("TF-IDF + Logistic Regression")

    st.write("**Queue Review Threshold:** 35%")

    st.write("**Priority Review Threshold:** 45%")

    st.divider()

    st.caption(
        "This system is designed as a decision-support "
        "copilot. Low-confidence predictions require "
        "human review."
    )


# -------------------------------------------------
# TICKET INPUT
# -------------------------------------------------

st.subheader("Support Ticket")

ticket_text = st.text_area(
    "Describe the user's issue",
    height=180,
    placeholder=(
        "Example: Our finance team cannot access the "
        "payment platform and several employees are "
        "unable to process transactions."
    )
)


analyze_button = st.button(
    "Analyze Ticket",
    type="primary",
    use_container_width=True
)


# -------------------------------------------------
# ANALYSIS
# -------------------------------------------------

if analyze_button:

    if not ticket_text.strip():

        st.warning(
            "Please enter a support ticket before analysis."
        )

    else:

        with st.spinner("Analyzing ticket..."):

            result = analyze_ticket(ticket_text)


        st.divider()

        st.subheader("Copilot Analysis")


        # Main result metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Recommended Queue",
                result["queue"]
            )

        with col2:
            st.metric(
                "Queue Confidence",
                f'{result["queue_confidence"]:.2f}%'
            )

        with col3:
            st.metric(
                "Predicted Priority",
                result["priority"].upper()
            )

        with col4:
            st.metric(
                "Priority Confidence",
                f'{result["priority_confidence"]:.2f}%'
            )


        st.divider()


        # Decision
        st.subheader("Decision")

        if result["decision"] == "HUMAN REVIEW REQUIRED":

            st.warning(
                "⚠️ HUMAN REVIEW REQUIRED — "
                "At least one prediction is below the "
                "configured confidence threshold."
            )

        else:

            st.success(
                "✅ AUTO-ROUTE ELIGIBLE — "
                "Predictions meet the configured "
                "confidence thresholds."
            )


               # Top queue predictions
        st.subheader("Top Routing Suggestions")

        for rank, item in enumerate(
            result["top_3_queues"],
            start=1
        ):

            queue, confidence = item

            st.write(
                f"**{rank}. {queue}** — "
                f"{confidence:.2f}%"
            )

            st.progress(
                min(confidence / 100, 1.0)
            )


                # ------------------------------------------
        # HISTORICAL RESOLUTION RECOMMENDATIONS
        # ------------------------------------------

        st.divider()

        st.subheader(
            "Historical Resolution Suggestions"
        )

        historical_matches = result[
            "historical_matches"
        ]

        if not historical_matches:

            st.info(
                "No sufficiently similar historical "
                "case was found. A technician should "
                "investigate the issue manually."
            )

        else:

            st.write(
                "The following suggestions come from "
                "similar previously resolved tickets."
            )

            for number, match in enumerate(
                historical_matches,
                start=1
            ):

                with st.expander(
                    f"Historical Match {number} — "
                    f"{match['similarity']:.2f}% similarity"
                ):

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write("**Queue**")
                        st.write(match["queue"])

                    with col2:
                        st.write("**Priority**")
                        st.write(
                            str(match["priority"]).upper()
                        )

                    with col3:
                        st.write("**Ticket Type**")
                        st.write(match["type"])

                    st.write(
                        "**Historical Ticket**"
                    )

                    st.write(
                        match["historical_ticket"]
                    )

                    st.write(
                        "**Reference Resolution from Historical Case**"
                    )

                    if match["similarity"] < 30:

                        st.warning(
                            "Similarity is relatively low. "
                            "Use this historical resolution only as a reference "
                            "and verify the steps before applying them."
                        )

                    st.info(
                        match["resolution"]
                    )

                    st.caption(
                        "Historical responses may contain customer names, "
                        "company wording, product-specific details, or placeholders. "
                        "Review and adapt the response before sending it to a user."
                    )


        st.divider()

        st.caption(
            "Predictions and historical resolutions are advisory. "
            "Low-confidence predictions and uncertain cases should "
            "be reviewed by support personnel."
        )