import streamlit as st
import pandas as pd
import base64
from orchestrator import Orchestrator
from langchain.callbacks.base import BaseCallbackHandler


# --- Streamlit page config ---
st.set_page_config(page_title="DQ Multi-Agent Orchestrator", layout="wide")

# --- Custom CSS forz̄ ChatGPT-style suggestion chips ---
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] button {
    border-radius: 20px;
    background-color: #f0f2f6;
    color: #333;
    border: 1px solid #ddd;
    padding: 6px 14px;
    margin: 4px;
    font-size: 14px;
}
div[data-testid="stHorizontalBlock"] button:hover {
    background-color: #e6eaf1;
    border-color: #999;
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
uploaded = st.sidebar.file_uploader("Upload Data File", type=["csv","xlsx"])

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logs" not in st.session_state:
    st.session_state.logs = {"thoughts": [], "tools": []}
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "df" not in st.session_state:
    st.session_state.df = None

# # --- Load dataset ---
# if uploaded is not None and st.session_state.df is None:
#     df = pd.read_csv(uploaded)
#     st.session_state.df = df
#     st.session_state.orchestrator = Orchestrator(df)
#     st.sidebar.success("✅ Dataset loaded.")


# --- Load dataset ---
if uploaded is not None and st.session_state.df is None:
    file_type = uploaded.name.split(".")[-1].lower()

    if file_type == "csv":
        df = pd.read_csv(uploaded)
    elif file_type in ["xlsx", "xls"]:
        df = pd.read_excel(uploaded)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        df = None

    if df is not None:
        st.session_state.df = df
        st.session_state.orchestrator = Orchestrator(df)
        st.sidebar.success("✅ Dataset loaded.")



# --- Main UI ---
st.title("🧠 Multi-Agent Data Quality Orchestrator")

# Show previous chat
for role, content in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(content)
    else:
        st.chat_message("assistant").write(content)

# --- Suggested Queries ---
st.markdown("### 💡 Suggested Queries")
suggested_queries = [
    "Profile this dataset",
    "Check data quality issues",
    "Generate business rules & fixes",
    "Suggest remediation",
    "Run full governance analysis"
]
cols = st.columns(len(suggested_queries))
clicked_query = None
for i, q in enumerate(suggested_queries):
    if cols[i].button(q):
        clicked_query = q


# --- Chat input ---
if st.session_state.orchestrator:
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("Or ask your own question:")
        submitted = st.form_submit_button("Send")

    if submitted or clicked_query:
        query = user_query if submitted else clicked_query
        st.session_state.messages.append(("user", query))
        st.chat_message("user").write(query)

        # Run orchestrator
        callback = StreamlitCallbackHandler()
        result = st.session_state.orchestrator.route(query, callbacks=[callback])

        # Save logs
        st.session_state.logs = callback.get_all_logs()

        # Show assistant reply
        assistant_box = st.chat_message("assistant")
        assistant_box.write(result.get("llm_response"))

        # Store message
        st.session_state.messages.append(("assistant", result.get("llm_response")))

        # Save tool outputs
        st.session_state.tool_outputs = result.get("tool_outputs", {})


# Agent Actions (stakeholder-friendly)
# --- Logs in tabs (bottom) ---
log_tabs = st.tabs(["🧩 Chain of Thought", "🤖 Agent Actions"])

# Chain of Thought (developer view)
with log_tabs[0]:
    st.markdown("## 🧩 Chain of Thought (Developer View)")
    for line in st.session_state.logs.get("thoughts", []):
        st.text(line)

# Agent Actions (stakeholder-friendly)
with log_tabs[1]:
    st.markdown("## 🤖 Agent Actions")
    tools = st.session_state.get("tool_outputs", {})
    if tools:
        for name, output in tools.items():
            # --- Special case: Profiling → only show dashboard ---
            if name.lower() == "profiling" and isinstance(output, dict):
                if "dashboards" in output:
                    for dash_name, dash_path in output["dashboards"].items():
                        try:
                            with open(dash_path, "r", encoding="utf-8") as f:
                                st.components.v1.html(f.read(), height=800, scrolling=True)
                        except Exception as e:
                            st.warning(f"Could not load {dash_name} dashboard: {e}")
                continue  # skip other stuff for profiling

            if name.lower() == "crossrelation" and isinstance(output, dict):
                st.markdown("### 🔗 Cross-Column Relationships")
                if not output["artifact"].empty:
                    st.dataframe(output["artifact"])
                else:
                    st.info("No inconsistencies detected across categorical relationships.")
                continue

            # --- Governance rendering ---
            if name.lower() == "governance" and isinstance(output, dict):
                st.markdown("### 🔧 Governance Results")

                if "artifact" in output:
                    st.markdown("**Per-column Governance Rules:**")
                    render_agent_output(output["artifact"])

                if "llm_summary" in output:
                    st.markdown("**Governance Summary:**")
                    st.markdown(output["llm_summary"])  # already bullet style

                if "dataset_rules" in output:
                    st.markdown("**Dataset-level Rules:**")
                    for rule in output["dataset_rules"]:
                        st.markdown(f"- {rule}")
                continue

            # --- Default rendering for other tools ---
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
                        except Exception as e:
                            st.warning(f"Could not load {dash_name} dashboard: {e}")
