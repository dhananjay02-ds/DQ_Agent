# llm_utils.py
# llm_utils.py
import os
import msal
import requests
from typing import Any, List, Optional, Union
import uuid

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage


# ========== Org-specific Config ==========
CLIENT_ID = os.getenv("MSAL_CLIENT_ID", "your-client-id")
TENANT_ID = os.getenv("MSAL_TENANT_ID", "your-tenant-id")
CLIENT_SECRET = os.getenv("MSAL_CLIENT_SECRET", "your-client-secret")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = ["api://your-gpt-api/.default"]

# GPT API endpoint (proxy your org provides)
GPT_ENDPOINT = os.getenv("GPT_ENDPOINT", "https://your-org-gpt-api.com/v1/chat")


# ========== Token Acquisition ==========
def _get_access_token() -> str:
    """Acquire token from Azure AD using MSAL Client Credentials Flow."""
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )
    token_result = app.acquire_token_for_client(scopes=SCOPES)
    if "access_token" not in token_result:
        raise Exception(f"MSAL failed: {token_result.get('error_description')}")
    return token_result["access_token"]


# ========== Custom LangChain Chat Model ==========
class MSALChatLLM(BaseChatModel):
    """LangChain-compatible wrapper for MSAL-authenticated GPT API (custom schema)."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, chat_type: str = "general"):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.chat_type = chat_type

    @property
    def _llm_type(self) -> str:
        return "msal-chat-llm"

    def _call_api(self, prompt: str, chat_id: Optional[str] = None) -> str:
        """Helper to call GPT REST API and return text, with robust parsing."""
        if chat_id is None:
            chat_id = str(uuid.uuid4())  # generate unique chat_id per session

        payload = {
            "prompt": prompt,
            "chat_id": chat_id,
            "prompt_prefix": "",          # optional: can inject system prompts here
            "chat_type": self.chat_type,
            "model": self.model,
            "temperature": self.temperature,
        }

        token = _get_access_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        resp = requests.post(GPT_ENDPOINT, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise Exception(f"GPT API error: {resp.status_code} {resp.text}")

        try:
            data = resp.json()
        except Exception as e:
            raise Exception(f"Failed to parse JSON response: {resp.text}") from e

        # --- Try common keys in response ---
        for key in ["content", "output", "text", "message", "response"]:
            if key in data and isinstance(data[key], str):
                return data[key]

        # --- Last fallback: dump entire JSON ---
        return str(data)

    # LangChain internal: agent/chain execution
    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs,
    ):
        # Flatten conversation into one prompt
        user_texts = [m.content for m in messages if m.type == "human"]
        prompt = "\n".join(user_texts)
        text = self._call_api(prompt)
        return [AIMessage(content=text)]

    # Direct `.invoke()` support
    def invoke(self, input: Union[str, List[dict]]) -> AIMessage:
        """
        For direct calls like: llm.invoke("Tell me a joke")
        Accepts either a plain string or a list of message dicts.
        """
        if isinstance(input, str):
            prompt = input
        else:
            # assume list of {role, content} dicts
            prompt = "\n".join([m["content"] for m in input if "content" in m])
        text = self._call_api(prompt)
        return AIMessage(content=text)


# ========== Factory Function ==========
def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0, chat_type: str = "general"):
    """
    Factory: by default use MSAL + GPT proxy.
    If USE_MSAL=false, fall back to OpenAI's ChatOpenAI.
    """
    if os.getenv("USE_MSAL", "true").lower() == "true":
        return MSALChatLLM(model=model, temperature=temperature, chat_type=chat_type)
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
