import streamlit as st
import pandas as pd
from orchestrator import Orchestrator
from llm_utils import get_llm
import io, base64
from plotly.graph_objs import Figure

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="🧠 Fidelity Copilot", layout="wide")

# -----------------------------
# Load Company Logo
# -----------------------------
try:
    with open("header.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="150" style="margin-right:15px; vertical-align:middle; border-radius:8px;">'
except FileNotFoundError:
    logo_html = ""

# -----------------------------
# Styles
# -----------------------------
st.markdown(
    """
    <style>
    body, div, p, span, input, textarea {
        font-family: "Inter", "Segoe UI", sans-serif !important;
    }

    /* --- Header --- */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding: 1rem 1.2rem;
        border-bottom: 1px solid #eaeaea;
        background-color: #ffffff;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        position: sticky;
        top: 0;
        z-index: 10;
    }
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #002663;
        margin: 0;
    }

    /* --- Section Titles --- */
    .section-title {
        font-size: 1.4rem;
        font-weight: 650;
        color: #002663;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .section-divider {
        height: 1px;
        background: #eaeaea;
        margin-bottom: 1rem;
    }

    /* --- Chat Feed --- */
    .scrollable-feed {
        max-height: 70vh;
        overflow-y: auto;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 16px;
        box-shadow: inset 0 0 8px rgba(0,0,0,0.03);
    }

    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .chat-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }
    .chat-left {
        justify-content: flex-start;
    }
    .chat-right {
        justify-content: flex-end;
    }

    .chat-bubble {
        padding: 0.9rem 1.1rem;
        border-radius: 18px;
        max-width: 70%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.08);
        line-height: 1.5;
        font-size: 0.95rem;
    }

    .user-bubble {
        background-color: #002663;
        color: white;
        border-top-right-radius: 4px;
    }

    .bot-bubble {
        background-color: #ffffff;
        color: #1a1a1a;
        border-top-left-radius: 4px;
        border: 1px solid #e0e0e0;
    }

    .chat-icon {
        font-size: 1.4rem;
        margin: 0 0.6rem;
    }

    /* --- Thought Box --- */
    .thought-box {
        background: #F8F9FA;
        border-left: 5px solid #002663;
        padding: 0.9rem 1rem;
        border-radius: 10px;
        color: #333;
        font-style: italic;
        margin-top: 0.4rem;
    }

    /* --- Chain Log --- */
    .log-box {
        background: #F4F6F8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #90CAF9;
        font-family: "Roboto Mono", monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
        color: #1A1A1A;
        margin-top: 0.6rem;
    }

    /* --- Sticky Input --- */
    .sticky-input {
        position: sticky;
        bottom: 0;
        background-color: #ffffff;
        padding: 1rem;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.06);
        border-top: 1px solid #e5e7eb;
        margin-top: 1.5rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Artifact Renderer
# -----------------------------
def render_agent_output(output):
    if output is None:
        return
    if isinstance(output, pd.DataFrame):
        st.dataframe(output.head(20))
        return
    if isinstance(output, Figure):
        st.plotly_chart(output, use_container_width=True)
        return
    if isinstance(output, str):
        st.markdown(output)
        return
    if isinstance(output, dict):
        if output.get("llm_summary"):
            st.markdown(f"**Summary:** {output['llm_summary']}")
        if "artifact" in output:
            render_agent_output(output["artifact"])
        if "table" in output:
            render_agent_output(output["table"])
        if "dashboards" in output and isinstance(output["dashboards"], dict):
            for name, path in output["dashboards"].items():
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        st.components.v1.html(fh.read(), height=480, scrolling=True)
                except Exception:
                    st.warning(f"Could not load dashboard {name}")
        if "image_base64" in output:
            st.image(base64.b64decode(output["image_base64"]), use_column_width=True)
        return
    st.write(output)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    f"""
    <div class='header-container'>
        {logo_html}
        <h1 class='app-title'>🧠 Fidelity Multi-Agent Data Quality Copilot</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("<div class='sidebar-title'>⚙️ Setup</div>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.conversation_history = []
    st.rerun()

# -----------------------------
# Session State
# -----------------------------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = None

# -----------------------------
# Dataset Section
# -----------------------------
st.markdown("<div class='section-title'>📂 Load Dataset</div>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith((".xlsx", ".xls")) else pd.read_csv(uploaded_file)
        st.session_state.df = df

        with st.expander("📊 Dataset Overview", expanded=False):
            st.dataframe(df.head())
            if st.button("🧠 Summarize Dataset"):
                with st.spinner("Generating summary..."):
                    llm = get_llm(model="gpt-4o-mini", temperature=0.0)
                    cols = df.columns.tolist()
                    sample = df.head(3).to_dict(orient="records")
                    prompt = (
                        f"Provide a concise summary of this dataset for a data quality analyst.\n"
                        f"Columns: {cols}\nSample rows: {sample}\n"
                        f"Summarize column types and data intent."
                    )
                    st.session_state.dataset_summary = llm.predict(prompt)
            if st.session_state.dataset_summary:
                st.markdown(f"#### 🧠 Quick Summary\n{st.session_state.dataset_summary}")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
else:
    st.info("📎 Please upload a dataset from the left sidebar to begin.")

# -----------------------------
# Chat History Section
# -----------------------------
st.markdown("<div class='section-title'>💬 Chat History</div>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='scrollable-feed'><div class='chat-container'>", unsafe_allow_html=True)

if len(st.session_state.conversation_history) == 0:
    st.markdown("*No conversation yet. Start by asking a question below.*")

for entry in st.session_state.conversation_history:
    # USER MESSAGE (RIGHT)
    st.markdown(
        f"""
        <div class="chat-row chat-right">
            <div class="chat-bubble user-bubble">
                👤 <strong>You</strong><br>{entry['user']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # BOT MESSAGE (LEFT)
    bot_text = (
        entry.get("llm_response")
        or entry.get("response")
        or "⚠️ No textual response returned by the Copilot."
    )
    st.markdown(
        f"""
        <div class="chat-row chat-left">
            <div class="chat-bubble bot-bubble">
                🤖 <strong>Copilot</strong><br>{bot_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Artifacts ---
    tool_outputs = entry.get("tool_outputs") or entry.get("artifacts", {})
    if tool_outputs:
        with st.expander("📦 Detailed Artifacts", expanded=False):
            for tool_name, output in tool_outputs.items():
                st.markdown(f"**🧩 {tool_name}**")
                render_agent_output(output)

    # --- Thought & Chain ---
    thoughts = entry.get("final_thought") or entry.get("thoughts")
    chain_trace = entry.get("chain_of_thought_log")
    if thoughts or chain_trace:
        with st.expander("💭 Chain of Thought", expanded=False):
            if thoughts:
                st.markdown(f"<div class='thought-box'>{thoughts}</div>", unsafe_allow_html=True)
            if chain_trace:
                st.markdown("##### 📜 Full Reasoning Trace")
                st.markdown(f"<div class='log-box'>{chain_trace}</div>", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

# -----------------------------
# Ask the Copilot Section
# -----------------------------
st.markdown("<div class='section-title'>🧠 Ask the Copilot</div>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='sticky-input'>", unsafe_allow_html=True)
query = st.text_area("💬 Type your query below", placeholder="e.g. Run a data quality assessment on fund type...", key="query_input")

if st.button("▶️ Run", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query.")
    elif st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        with st.spinner("Running orchestrator..."):
            orchestrator = Orchestrator(st.session_state.df)
            result = orchestrator.route(query)

            st.session_state.conversation_history.append({
                "user": query,
                "llm_response": result.get("llm_response", ""),
                "final_thought": result.get("final_thought", ""),
                "tool_outputs": result.get("tool_outputs", {}),
                "chain_of_thought_log": result.get("chain_of_thought_log", "")
            })
            st.session_state.clear_input = True
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True)
