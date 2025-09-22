# agents/rule_generation.py
from typing import Dict, Any
from llm_utils import get_llm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class RuleGenerationAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = get_llm(model=model)
        self.prompt = PromptTemplate(
            input_variables=["context"],
            template=(
                "You are a data analyst that converts dataset profiling and DQ scoring results into"
                " two sets of business rules: (A) Inferred rules (learned from data patterns) and"
                " (B) Observed rules (expected constraints based on violations). Given the context:\n\n"
                "{context}\n\n"
                "Return a JSON-like structure with two keys: inferred_rules (list) and observed_rules (list). "
                "Each rule should be short (one line) and tagged with rationale."
            )
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def run(self, df=None, query: str = None, context: str = None) -> Dict[str, Any]:
        ctx_text = context or "No additional context provided."
        resp = self.chain.run({"context": ctx_text})
        # We return the raw LLM response and save to outputs/rules.txt
        import os, json
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", "generated_rules.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(resp)
        return {"rules_path": out_path, "rules_text": resp}

rule_generation_agent = RuleGenerationAgent()
