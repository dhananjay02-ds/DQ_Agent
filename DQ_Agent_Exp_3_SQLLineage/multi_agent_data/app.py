import streamlit as st
import pandas as pd
import base64
from orchestrator import Orchestrator
from langchain.callbacks.base import BaseCallbackHandler


# --- Streamlit page config ---
st.set_page_config(page_title="DQ Multi-Agent Orchestrator", layout="wide")

# --- Custom CSS for bubble-style buttons ---
st.markdown("""
<style>
button[kind="secondary"] {
    border-radius: 20px !important;
    background-color: #f0f2f6 !important;
    color: #333 !important;
    border: 1px solid #ddd !important;
    padding: 6px 14px !important;
    margin: 4px !important;
    font-size: 14px !important;
}
button[kind="secondary"]:hover {
    background-color: #e6eaf1 !important;
    border-color: #999 !important;
}
div.chat-history {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 12px;
    margin: 12px 0;
    max-height: 400px;
    overflow-y: auto;
    background-color: #fafafa;
}
</style>
""", unsafe_allow_html=True)


# --- Callback Handler for reasoning logs ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs = []
        self.tool_logs = []

    def on_agent_action(self, action, **kwargs):
        self.logs.append(f"ACTION: {action}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "tool")
        self.tool_logs.append(f"TOOL START: {name} | input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        self.tool_logs.append(f"TOOL END: {str(output)[:200]}")

    def on_llm_start(self, *args, **kwargs):
        self.logs.append("🤖 LLM START")

    def on_llm_end(self, response, **kwargs):
        txt = None
        try:
            txt = response.generations[0][0].text
        except Exception:
            try:
                txt = getattr(response, "content", str(response))[:400]
            except Exception:
                txt = str(response)[:400]
        self.logs.append(f"🤖 LLM END: {txt}")

    def get_all_logs(self):
        return {"thoughts": self.logs, "tools": self.tool_logs}


# --- Renderer for agent artifacts ---
def render_agent_output(output):
    if output is None:
        return
    if isinstance(output, pd.DataFrame):
        st.dataframe(output.head(20))
    elif isinstance(output, str):
        if output.endswith(".csv"):
            st.download_button("⬇️ Download CSV", open(output, "rb"), file_name="result.csv")
            try:
                df = pd.read_csv(output)
                st.dataframe(df.head(20))
            except Exception:
                st.text(open(output).read()[:1000])
        else:
            st.text(output[:1000])
    elif isinstance(output, dict):
        if "image_base64" in output:
            st.image(base64.b64decode(output["image_base64"]), use_column_width=True)


# --- Sidebar ---
st.sidebar.title("⚙️ Setup")
uploaded = st.sidebar.file_uploader("Upload Data File", type=["csv", "xlsx"])

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logs" not in st.session_state:
    st.session_state.logs = {"thoughts": [], "tools": []}
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "df" not in st.session_state:
    st.session_state.df = None

# --- Always ensure orchestrator exists (for SQL-only flow) ---
if st.session_state.orchestrator is None:
    st.session_state.orchestrator = Orchestrator(df=None)

# --- Load dataset ---
if uploaded is not None and st.session_state.df is None:
    file_type = uploaded.name.split(".")[-1].lower()
    df = None
    if file_type == "csv":
        df = pd.read_csv(uploaded)
    elif file_type in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded)
    if df is not None:
        st.session_state.df = df
        st.session_state.orchestrator = Orchestrator(df)   # ✅ replaces df=None orchestrator
        st.sidebar.success("✅ Dataset loaded.")

# --- Main UI ---
st.title("🧠 Multi-Agent Data Quality Orchestrator")

# --- KPI Cards ---
if st.session_state.df is not None:
    cols = st.columns(3)
    with cols[0]:
        st.metric("Rows", len(st.session_state.df))
    with cols[1]:
        st.metric("Columns", len(st.session_state.df.columns))
    with cols[2]:
        dq_score = None
        if hasattr(st.session_state, "tool_outputs") and "Advanced Profiling" in st.session_state.tool_outputs:
            dq_score = st.session_state.tool_outputs["Advanced Profiling"].get("dataset_score")
        st.metric("DQ Score", dq_score if dq_score else "--")

# --- Dataset Preview (collapsible) ---
if st.session_state.df is not None:
    with st.expander("📊 Preview Dataset", expanded=False):
        st.dataframe(st.session_state.df.head(10))

# --- SQL Query Input ---
st.markdown("### 📝 SQL Query (for lineage analysis)")
with st.form(key="sql_form", clear_on_submit=True):
    sql_query = st.text_area("Paste your SQL query here:")
    sql_submitted = st.form_submit_button("Run SQL Lineage")

if sql_submitted and sql_query:
    query = f"Lineage Analysis: {sql_query}"
    st.session_state.messages.append(("user", query))

    # Run orchestrator
    callback = StreamlitCallbackHandler()
    result = st.session_state.orchestrator.route(query, callbacks=[callback])

    # Save logs
    st.session_state.logs = callback.get_all_logs()

    # Store assistant reply
    st.session_state.messages.append(("assistant", result.get("llm_response")))

    # Save tool outputs
    st.session_state.tool_outputs = result.get("tool_outputs", {})


# --- Suggested Queries ---
st.markdown("### 💡 Suggested Queries")
suggested_queries = [
    "Give me a summary profile of this dataset",          # ProfilingAgent
    "What is wrong with my data?",                        # AdvancedProfiling + Insights
    "What business rules should govern this dataset?",    # GovernanceAgent
    "Are the relationships between columns consistent?",  # CrossRelationAgent
    "Show me the top 5 funds with the highest returns",   # Text2DFAgent
    "Visualize the distribution of returns",              # EDAAgent
    "Which tables are used in this SQL query?",           # LineageAgent
    "Explain lineage of this SQL query"                   # LineageAgent
]

clicked_query = None
cols = st.columns(len(suggested_queries))
for i, q in enumerate(suggested_queries):
    with cols[i]:
        if st.button(q, key=f"suggested_{i}"):
            clicked_query = q

# --- Chat input ---
if st.session_state.orchestrator:
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("💬 Type your question:")
        submitted = st.form_submit_button("Send")

    if submitted or clicked_query:
        if submitted:
            query = user_query
        elif clicked_query:
            query = clicked_query
        else:
            query = ""

        st.session_state.messages.append(("user", query))

        # Run orchestrator
        callback = StreamlitCallbackHandler()
        result = st.session_state.orchestrator.route(query, callbacks=[callback])

        # Save logs
        st.session_state.logs = callback.get_all_logs()

        # Store assistant reply
        st.session_state.messages.append(("assistant", result.get("llm_response")))

        # Save tool outputs
        st.session_state.tool_outputs = result.get("tool_outputs", {})

# --- Chat History Box ---
if st.session_state.messages:
    st.markdown("### 💬 Conversation")
    st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
    for role, content in st.session_state.messages:
        if role == "user":
            st.markdown(f"**🧑 You:** {content}")
        else:
            st.markdown(f"**🤖 Assistant:** {content}")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Agent Outputs ---
st.markdown("## 🔧 Agent Outputs")
tools = st.session_state.get("tool_outputs", {})
if tools:
    for name, output in tools.items():
        # Profiling dashboard
        if name.lower() == "profiling" and isinstance(output, dict):
            if "dashboards" in output:
                for dash_name, dash_path in output["dashboards"].items():
                    try:
                        with open(dash_path, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=800, scrolling=True)
                        st.download_button(
                            label=f"⬇️ Download {dash_name.capitalize()} Report",
                            data=open(dash_path, "rb"),
                            file_name=f"{dash_name}_report.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.warning(f"Could not load {dash_name} dashboard: {e}")
            continue

        # CrossRelation rendering
        if name.lower() == "crossrelation" and isinstance(output, dict):
            st.markdown("### 🔗 Cross-Column Relationships")
            if not output["artifact"].empty:
                st.dataframe(output["artifact"])
            else:
                st.info("No inconsistencies detected across categorical relationships.")
            continue

        # Governance rendering
        if name.lower() == "governance" and isinstance(output, dict):
            st.markdown("### 🛠 Governance Results")
            if "artifact" in output:
                st.markdown("**Per-column Governance Rules:**")
                render_agent_output(output["artifact"])
            if "llm_summary" in output:
                st.markdown("**Governance Summary:**")
                st.markdown(output["llm_summary"])
            if "dataset_rules" in output:
                st.markdown("**Dataset-level Rules:**")
                for rule in output["dataset_rules"]:
                    st.markdown(f"- {rule}")
            continue

        # Lineage rendering
        if name.lower() == "lineage" and isinstance(output, dict):
            st.markdown("### 🧭 Lineage Results")
            if "llm_summary" in output:
                st.markdown(f"**Explanation:** {output['llm_summary']}")
            if "artifact" in output:
                st.markdown("**Lineage JSON:**")
                st.json(output["artifact"])
            continue

        # Default rendering
        st.markdown(f"### 🔧 {name}")
        if isinstance(output, dict):
            if "llm_summary" in output:
                st.markdown(f"**Summary:** {output['llm_summary']}")
            if "artifact" in output:
                st.markdown("**Artifact Table:**")
                render_agent_output(output["artifact"])
            if "dashboards" in output:
                for dash_name, dash_path in output["dashboards"].items():
                    try:
                        with open(dash_path, "r", encoding="utf-8") as f:
                            st.markdown(f"**{dash_name.capitalize()} Dashboard:**")
                            st.components.v1.html(f.read(), height=800, scrolling=True)
                        st.download_button(
                            label=f"⬇️ Download {dash_name.capitalize()} Report",
                            data=open(dash_path, "rb"),
                            file_name=f"{dash_name}_report.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.warning(f"Could not load {dash_name} dashboard: {e}")

# --- Logs ---
log_tabs = st.tabs(["🧩 Thought Process (dev)", "📜 Tool Logs"])
with log_tabs[0]:
    st.markdown("## 🧩 LLM Reasoning Trace")
    for line in st.session_state.logs.get("thoughts", []):
        st.text(line)
with log_tabs[1]:
    st.markdown("## 📜 Tool Logs")
    for line in st.session_state.logs.get("tools", []):
        st.text(line)
