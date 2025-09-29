# llm_utils.py
import os
from langchain_openai import ChatOpenAI

def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """
    Returns a ChatOpenAI instance (LangChain wrapper).
    Requires OPENAI_API_KEY environment variable to be set.
    """
    os.environ["OPENAI_API_KEY"] = ""
    
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set in environment")
    return ChatOpenAI(model=model, temperature=temperature)
