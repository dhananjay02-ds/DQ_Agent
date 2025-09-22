# app.py
import streamlit as st
import pandas as pd
import uuid
import os

from orchestrator import Orchestrator
from agents.eda import eda_agent
from agents.profiling import profiling_agent
from agents.dq_scoring import dq_scoring_agent

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# --- SESSION HANDLING ---
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.set_page_config(page_title="AI Data Quality Assistant", layout="wide")

st.title("🤖 AI Data Quality Copilot")
st.caption(f"Session ID: `{st.session_state['session_id']}`")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
    st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Show preview
    with st.expander("📊 Preview Dataset"):
        st.dataframe(df.head())

    # Create orchestrator instance
    if "orchestrator" not in st.session_state:
        st.session_state["orchestrator"] = Orchestrator(df)

    orchestrator = st.session_state["orchestrator"]

    # --- CHAT INTERFACE ---
    st.subheader("💬 Chat with Data Quality Assistant")
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask me about data quality, profiling, rules, or run a query")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        # Append user message
        st.session_state["messages"].append(("user", user_input))

        # Run orchestrator
        try:
            result = orchestrator.route(user_input)
        except Exception as e:
            result = f"⚠️ Error: {e}"

        # Append assistant response
        st.session_state["messages"].append(("assistant", result))

    # --- DISPLAY CHAT HISTORY ---
    for role, content in st.session_state["messages"]:
        if role == "user":
            st.markdown(f"**🧑 You:** {content}")
        else:
            st.markdown(f"**🤖 Assistant:** {content}")

    # --- EXTRA ACTION BUTTONS ---
    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Run Profiling"):
            profiling_res = profiling_agent.run(df)
            st.dataframe(profiling_res["summary_df"])
            st.session_state["messages"].append(("assistant", "📊 Profiling report generated."))
    
    with col2:
        if st.button("Run DQ Scoring"):
            scoring_res = dq_scoring_agent.run(df)  # ✅ direct agent call
            st.write("### 🧹 Data Quality Scoring")
            st.dataframe(scoring_res["dq_df"])
            st.session_state["messages"].append(("assistant", scoring_res["summary_text"]))
    
    with col3:
        if st.button("Plot Correlations"):
            import matplotlib.pyplot as plt
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
            st.session_state["messages"].append(("assistant", "📈 Correlation heatmap plotted."))

else:
    st.info("Please upload a CSV dataset to begin.")
