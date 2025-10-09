# app.py (robust cross-platform header + chat UI with chat history sections + improved header image alignment)
import streamlit as st
import pandas as pd
import base64
from orchestrator import Orchestrator
from langchain.callbacks.base import BaseCallbackHandler
from plotly.graph_objs import Figure

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="🧠 DQ Copilot", layout="wide")

# -------------------------
# Safe CSS styling (cross-platform)
# -------------------------
st.markdown(
    """
<style>
body, div, p, span, input, textarea {
  font-family: Inter, 'Segoe UI', sans-serif !important;
}

/* improved header alignment */
.header-container {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 16px;
  margin-bottom: 0.4rem;
}

/* suggestions row */
.suggestion-row {
  display:flex;
  gap:12px;
  align-items:center;
  justify-content:center;
  margin-bottom: 8px;
}

/* chat containers */
.chat-container {
  background: #fafafa;
  border-radius: 12px;
  padding: 12px;
  height: 56vh;
  overflow: hidden;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  margin-top: 8px;
}

.scrollable-chat {
  height: 100%;
  overflow-y: auto;
  padding-right: 8px;
}

/* chat history title blocks */
.chat-history-title {
  font-weight: 700;
  color: #374151;
  font-size: 1.05rem;
  margin-top: 0.4rem;
  margin-bottom: 0.4rem;
  border-bottom: 1px solid #e5e7eb;
  padding-bottom: 4px;
}

.chat-session-title {
  font-weight: 700;
  color: #2563eb;
  font-size: 1.05rem;
  margin-top: 1.2rem;
  margin-bottom: 0.4rem;
  border-bottom: 1px solid #c7d2fe;
  padding-bottom: 4px;
}

/* visual grouping */
.chat-history-container {
  background: #f9fafb;
  border-radius: 12px;
  padding: 0.6rem 1rem;
  margin-bottom: 0.8rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.current-chat-container {
  background: #ffffff;
  border-radius: 12px;
  padding: 0.8rem 1rem;
  box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}

/* chat bubbles */
.chat-bubble {
  display:flex;
  align-items:flex-start;
  gap:8px;
  margin:8px 0;
}
.chat-bubble .avatar { font-size:1.05rem; margin-top:4px; }
.chat-bubble .message {
  padding:8px 12px;
  border-radius:12px;
  line-height:1.4;
  max-width:86%;
  word-break:break-word;
  white-space:normal;
}
.chat-bubble.user .message { background:#dbeafe; color:#111; }
.chat-bubble.assistant .message { background:#f4f4f5; color:#111; }

/* artifacts (tool cards) */
.tool-card {
  background: #fff;
  border-radius: 10px;
  padding: 10px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.04);
  margin: 10px 0 14px 44px;
}

/* final sections */
.final-thought { background:#f7fafc; padding:10px; border-radius:8px; margin-top:8px; }
.final-answer { background:#ecfdf5; padding:10px; border-left:4px solid #10b981; border-radius:8px; margin-top:8px; }

/* suggestion buttons */
.suggestion-btn {
  background: #fff;
  border-radius: 12px;
  border: 1px solid #e6e6e6;
  padding: 10px 18px;
  cursor: pointer;
}
.suggestion-btn:hover { background:#f4f7fb; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Callback handler (logs)
# -------------------------
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs, self.tool_logs = [], []

    def on_agent_action(self, action, **kwargs):
        self.logs.append(f"ACTION: {action}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "tool") if isinstance(serialized, dict) else getattr(serialized, "name", "tool")
        self.tool_logs.append(f"TOOL START: {name} | input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        self.tool_logs.append(f"TOOL END: {str(output)[:200]}")

    def get_all_logs(self):
        return {"thoughts": self.logs, "tools": self.tool_logs}


# -------------------------
# Artifact renderer
# -------------------------
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
        if output.lower().endswith(".csv"):
            try:
                with open(output, "rb") as fh:
                    data = fh.read()
                st.download_button("⬇️ Download CSV", data, file_name="result.csv")
            except Exception:
                pass
        st.text(output[:2000])
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


# -------------------------
# Session state initialization
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "artifacts" not in st.session_state:
    st.session_state.artifacts = []
if "logs" not in st.session_state:
    st.session_state.logs = {"thoughts": [], "tools": []}
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "df" not in st.session_state:
    st.session_state.df = None


# -------------------------
# Sidebar setup
# -------------------------
st.sidebar.title("⚙️ Setup")
uploaded = st.sidebar.file_uploader("Upload Data File", type=["csv", "xlsx"])

if uploaded is not None and st.session_state.df is None:
    ext = uploaded.name.split(".")[-1].lower()
    try:
        df = pd.read_csv(uploaded) if ext == "csv" else pd.read_excel(uploaded)
        st.session_state.df = df
        st.session_state.orchestrator = Orchestrator(df)
        st.sidebar.success("✅ Dataset loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")


# -------------------------
# Header (left title + closer right image)
# -------------------------
try:
    with open("header_icon.png", "rb") as f:
        header_image_b64 = base64.b64encode(f.read()).decode()
except FileNotFoundError:
    header_image_b64 = None

header_html = f"""
<div class="header-container">
  <div>
    <h1 style='font-size:1.8rem; font-weight:700; margin:0;'>🧠 Multi-Agent Data Quality Copilot</h1>
    <p style='color:#666; font-size:0.95rem; margin:2px 0 0 0;'>
      Ask questions about your dataset — agents will analyze, validate, and summarize automatically.
    </p>
  </div>
  <div>
    {'<img src="data:image/png;base64,' + header_image_b64 + '" width="512" style="border-radius:10px; margin-left:24px;margin-top:-10px">' if header_image_b64 else ''}
  </div>
</div>
"""

st.markdown(header_html, unsafe_allow_html=True)
st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)


# -------------------------
# Dataset preview
# -------------------------
if st.session_state.df is not None:
    with st.expander("📊 Preview dataset", expanded=False):
        df = st.session_state.df
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10))
        st.markdown(f"**Columns:** {', '.join(df.columns)}")


# -------------------------
# Suggestion buttons
# -------------------------
suggestions = [
    "Profile this dataset",
    "Check data quality issues",
    "Generate business rules & fixes",
    "Suggest remediation",
    "Run full governance analysis",
]

cols = st.columns(len(suggestions))
clicked_suggestion = None
for i, s in enumerate(suggestions):
    if cols[i].button(s):
        clicked_suggestion = s


# -------------------------
# Chat container with clear separation
# -------------------------
st.markdown('<div class="chat-container"><div class="scrollable-chat">', unsafe_allow_html=True)

# 🗂 Chat History
if len(st.session_state.messages) > 1:
    st.markdown('<div class="chat-history-title">🗂 Chat History</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.messages[:-1]):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        avatar = "🧑‍💼" if role == "user" else "🤖"
        css_class = "user" if role == "user" else "assistant"
        st.markdown(
            f"""
            <div class="chat-bubble {css_class}">
              <div class="avatar">{avatar}</div>
              <div class="message">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# 💬 Current Conversation
st.markdown('<div class="chat-session-title">💬 Current Conversation</div>', unsafe_allow_html=True)
st.markdown('<div class="current-chat-container">', unsafe_allow_html=True)

if st.session_state.messages:
    last_idx = len(st.session_state.messages) - 1
    last_msg = st.session_state.messages[-1]
    role = last_msg.get("role", "assistant")
    content = last_msg.get("content", "")
    avatar = "🧑‍💼" if role == "user" else "🤖"
    css_class = "user" if role == "user" else "assistant"
    st.markdown(
        f"""
        <div class="chat-bubble {css_class}">
          <div class="avatar">{avatar}</div>
          <div class="message">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for art in st.session_state.artifacts:
        if art.get("parent_idx") == last_idx:
            st.markdown(f"<div class='tool-card'><strong>🧩 {art.get('tool')}</strong></div>", unsafe_allow_html=True)
            if art.get("summary"):
                st.markdown(f"**Summary:** {art.get('summary')}")
            render_agent_output(art.get("artifact"))

st.markdown("</div></div>", unsafe_allow_html=True)


# -------------------------
# Input area
# -------------------------
if st.session_state.orchestrator:
    user_query = st.text_input("💬 Ask a question:", key="user_input", placeholder="e.g., Check data quality issues", label_visibility="collapsed")
    send = st.button("Send")

    if send or clicked_suggestion:
        query = clicked_suggestion if clicked_suggestion else user_query
        if not query or not str(query).strip():
            st.warning("Please enter a query.")
        else:
            st.session_state.messages.append({"role": "user", "content": str(query)})
            callback = StreamlitCallbackHandler()

            with st.spinner("🧩 Agents analyzing..."):
                result = st.session_state.orchestrator.route(query, callbacks=[callback])

            st.session_state.logs = callback.get_all_logs()

            final_answer = result.get("llm_response", "") or ""
            final_thought = result.get("final_thought", "") or ""
            assistant_html = (
                f"<div><strong>✅ Final Answer:</strong><br>{final_answer}</div>"
                f"<div style='margin-top:8px;'><strong>💭 Thought:</strong><br>{final_thought}</div>"
            )

            st.session_state.messages.append({"role": "assistant", "content": assistant_html})
            assistant_idx = len(st.session_state.messages) - 1

            tool_outputs = result.get("tool_outputs", {}) or {}
            for tool_name, out in tool_outputs.items():
                summary = None
                artifact_obj = None
                if isinstance(out, dict):
                    summary = out.get("llm_summary") or out.get("summary")
                    artifact_obj = out.get("artifact") if "artifact" in out else None
                    if artifact_obj is None and not summary:
                        artifact_obj = out
                else:
                    artifact_obj = out

                st.session_state.artifacts.append(
                    {"parent_idx": assistant_idx, "tool": tool_name, "summary": summary, "artifact": artifact_obj}
                )

            st.markdown(
                f"""
                <div class="chat-bubble assistant">
                  <div class="avatar">🤖</div>
                  <div class="message">{assistant_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for art in st.session_state.artifacts:
                if art.get("parent_idx") == assistant_idx:
                    st.markdown(f"<div class='tool-card'><strong>🧩 {art.get('tool')}</strong></div>", unsafe_allow_html=True)
                    if art.get("summary"):
                        st.markdown(f"**Summary:** {art.get("summary")}")
                    render_agent_output(art.get("artifact"))


# -------------------------
# Developer logs
# -------------------------
with st.sidebar.expander("🧠 Developer Logs"):
    st.markdown("### Chain of Thought")
    for line in st.session_state.logs.get("thoughts", []):
        st.text(line)
    st.markdown("### Tool Logs")
    for line in st.session_state.logs.get("tools", []):
        st.text(line)
