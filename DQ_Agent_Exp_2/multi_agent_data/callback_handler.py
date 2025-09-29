# callback_handler.py
from langchain.callbacks.base import BaseCallbackHandler


def _extract_llm_text(response) -> str:
    try:
        if hasattr(response, "generations"):
            gens = response.generations
            if gens and hasattr(gens[0][0], "text"):
                return gens[0][0].text
        if hasattr(response, "text"):
            return response.text
        if hasattr(response, "content"):
            return response.content
        return str(response)
    except Exception:
        return str(response)


class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs = []
        self.tool_logs = []

    def on_agent_action(self, action, **kwargs):
        self.logs.append(f"ACTION: {action}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "tool") if isinstance(serialized, dict) else "tool"
        self.tool_logs.append(f"TOOL START: {name} | input: {str(input_str)[:200]}")

    def on_tool_end(self, output, **kwargs):
        self.tool_logs.append(f"TOOL END: {str(output)[:200]}")

    def on_llm_end(self, response, **kwargs):
        self.logs.append(f"🤖 LLM END: {_extract_llm_text(response)}")

    def get_all_logs(self):
        return {"thoughts": self.logs, "tools": self.tool_logs}
