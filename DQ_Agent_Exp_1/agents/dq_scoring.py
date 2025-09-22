# agents/dq_scoring.py
# agents/dq_scoring.py
import os
import json
import pandas as pd
from typing import Dict, Any
from llm_utils import get_llm

os.makedirs("outputs", exist_ok=True)

class DQScoringAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = get_llm(model=model)

    def _analyze_column(self, col: str, s: pd.Series) -> Dict[str, Any]:
        dtype = str(s.dtype)
        pct_missing = s.isna().mean() * 100
        unique_count = s.nunique(dropna=True)
        sample_values = s.dropna().astype(str).sample(
            min(10, len(s.dropna())), random_state=42
        ).tolist() if len(s.dropna()) > 0 else []
    
        prompt = f"""
        You are a Data Quality Analyst.
        Column: {col}
        Dtype: {dtype}
        Missing: {pct_missing:.2f}%
        Unique Values: {unique_count}
        Sample Values: {sample_values}
    
        Task:
        - Identify possible data quality issues (e.g., missing values, duplicates, invalid ranges, outliers, format errors).
        - Estimate severity (% of records affected) if possible.
        - Assign an overall data quality score (0–100, higher = better).
        - Output strictly as JSON in this format:
        {{
          "score": int,
          "issues": ["issue1", "issue2", ...]
        }}
        """
    
        response = self.llm.predict(prompt).strip()
    
        # ✅ Clean common wrappers before parsing
        if response.startswith("```"):
            response = response.strip("`")              # remove all backticks
            response = response.replace("json", "", 1)  # drop 'json' after ``` if present
    
        try:
            parsed = json.loads(response)
        except Exception as e:
            parsed = {"score": 100, "issues": [f"LLM parse error: {str(e)} | raw: {response}"]}
    
        return parsed


    def run(self, df: pd.DataFrame, query: str = None, target_col: str = None) -> Dict[str, Any]:
        rows = []
        for col in df.columns:
            result = self._analyze_column(col, df[col])
            rows.append({
                "Column": col,
                "DQ_Score": result.get("score", 100),
                "Issues": "; ".join(result.get("issues", [])) if result.get("issues") else "No major issues"
            })

        out_df = pd.DataFrame(rows).sort_values("DQ_Score")
        out_path = os.path.join("outputs", "dq_scoring.csv")
        out_df.to_csv(out_path, index=False)

        return {
            "dq_path": out_path,
            "dq_df": out_df,
            "summary_text": f"[DQScoringAgent] Results saved to {out_path}"
        }

dq_scoring_agent = DQScoringAgent()
