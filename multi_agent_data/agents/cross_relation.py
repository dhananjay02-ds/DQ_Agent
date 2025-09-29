import pandas as pd
from llm_utils import get_llm

class CrossRelationAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.llm = get_llm(model=model)

    def discover_relationships(self, df: pd.DataFrame, max_cardinality=50):
        """Find candidate categorical pairs."""
        cats = [c for c in df.columns if df[c].nunique() <= max_cardinality]
        pairs = [(c1, c2) for i, c1 in enumerate(cats) for c2 in cats[i+1:]]
        return pairs

    def filter_with_llm(self, df: pd.DataFrame, pairs):
        """Ask LLM which pairs matter semantically."""
        sample_values = {c: df[c].dropna().unique()[:5].tolist() for c in df.columns}
        prompt = f"""
        You are a data quality analyst.

        Columns and sample values:
        {sample_values}

        Candidate pairs: {pairs}

        Task:
        - Identify which pairs are meaningful relationships (e.g., FundName ↔ FundType).
        - Output JSON strictly:
          {{"relationships": [{{"col1": "...", "col2": "...", "reason": "..."}}]}}
        """
        response = self.llm.predict(prompt)

        # --- basic JSON parse (robust parsing can be added) ---
        import json
        try:
            parsed = json.loads(response)
            return parsed.get("relationships", [])
        except Exception:
            return []

    def check_consistency(self, df: pd.DataFrame, col1, col2, threshold=0.8):
        """Check mapping consistency between two columns."""
        mapping = df.groupby(col1)[col2].nunique()
        inconsistent = mapping[mapping > 1]
        return inconsistent

    def run(self, df: pd.DataFrame):
        pairs = self.discover_relationships(df)
        relationships = self.filter_with_llm(df, pairs)
        results = []

        for rel in relationships:
            inconsistent = self.check_consistency(df, rel["col1"], rel["col2"])
            if not inconsistent.empty:
                results.append({
                    "relationship": rel,
                    "inconsistencies": inconsistent.to_dict()
                })

        return {
            "llm_summary": relationships,
            "artifact": pd.DataFrame(results) if results else pd.DataFrame()
        }
