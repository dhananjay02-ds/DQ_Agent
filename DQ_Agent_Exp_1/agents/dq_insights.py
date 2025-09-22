# agents/dq_insights.py
import os
import re
from typing import Dict, Any
from llm_utils import get_llm

os.makedirs("outputs", exist_ok=True)

class DQInsightsAgent:
    def __init__(self, eda_agent, model: str = "gpt-4o-mini"):
        self.eda_agent = eda_agent
        self.llm = get_llm(model=model)

    def _extract_columns_list(self, llm_text: str):
        match = re.search(r"COLUMNS\s*=\s*\[([^\]]+)\]", llm_text)
        if not match:
            # try simple comma separated line starting with columns:
            m2 = re.search(r"columns\s*:\s*(.+)", llm_text, flags=re.IGNORECASE)
            if m2:
                cols = [c.strip(" []'\"") for c in m2.group(1).split(",")]
                return [c for c in cols if c]
            return []
        cols = [c.strip().strip("'\"") for c in match.group(1).split(",")]
        return [c for c in cols if c]

    def run(self, df: 'pd.DataFrame' = None, query: str = None, context: str = None, top_n: int = 3) -> Dict[str, Any]:
        """
        context is expected to be concatenation of profiling text + dq scoring text.
        Returns dict with keys: insights_text, eda_results (dict col -> eda output)
        """
        if context is None:
            context = "No context provided."

        prompt = f"""
You are a Data Quality Analyst. Given the profiling and DQ scoring context below:
{context}

Task:
1) Identify the top {top_n} columns with the lowest quality.
2) For each column, describe the issue and give 2-3 concrete remediation steps (with pandas code hints if possible).
3) Output a line that lists columns the assistant should plot in this exact format:
COLUMNS=[col1, col2, col3]

Provide the response in clear sections.
"""
        llm_resp = self.llm.predict(prompt)
        cols = self._extract_columns_list(llm_resp)
        eda_results = {}
        # trigger EDA for each column found (limit top_n)
        if df is not None and cols:
            for c in cols[:top_n]:
                if c in df.columns:
                    eda_results[c] = self.eda_agent.run(df, target_col=c)

        # save insights
        out_path = os.path.join("outputs", "dq_insights.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(llm_resp)

        return {"insights_path": out_path, "insights_text": llm_resp, "eda_results": eda_results}


