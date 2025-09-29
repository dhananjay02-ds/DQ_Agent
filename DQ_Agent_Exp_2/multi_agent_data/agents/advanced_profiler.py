import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from llm_utils import get_llm
import re

class AdvancedProfilingAgent:
    def __init__(self, model="gpt-4o-mini", baseline: pd.DataFrame = None):
        self.llm = get_llm(model=model)
        self.baseline = baseline

    def _schema_profile(self, df):
        return [{"Column": c, "Type": str(df[c].dtype)} for c in df.columns]

    def _statistical_profile(self, s):
        stats = {
            "Null %": 100 * s.isna().mean(),
            "Unique Count": s.nunique(dropna=True),
        }
        if pd.api.types.is_numeric_dtype(s):
            stats.update({
                "Mean": s.mean(),
                "Std": s.std(),
                "Min": s.min(),
                "Max": s.max(),
            })
        elif pd.api.types.is_datetime64_any_dtype(s):
            stats["Max Date"] = s.max()
            stats["Min Date"] = s.min()
            stats["Freshness (days)"] = (pd.Timestamp.today() - s.max()).days
        return stats

    def _uniqueness(self, s):
        return 100 * (s.nunique() / len(s)) if len(s) else 0

    def _outliers(self, s):
        if not pd.api.types.is_numeric_dtype(s) or len(s.dropna()) < 10:
            return 0
        z_outliers = (np.abs(zscore(s.dropna())) > 3).mean() * 100
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(s.dropna().to_frame())
        iso_outliers = (preds == -1).mean() * 100
        return max(z_outliers, iso_outliers)

    def _detect_id_column(self, col_name, s):
        """
        Heuristic to detect ID/index-like columns.
        """
        if re.search(r"(id|code|number)$", col_name.lower()):
            if s.nunique() == len(s):  # fully unique
                return True
        return False

    def run(self, df: pd.DataFrame):
        results = []
        for col in df.columns:
            s = df[col]

            col_res = {"Column": col, "Type": str(s.dtype)}
            col_res.update(self._statistical_profile(s))
            col_res["Uniqueness %"] = self._uniqueness(s)
            col_res["Outlier %"] = self._outliers(s)
            col_res["Is_ID"] = self._detect_id_column(col, s)

            # Simple deterministic scoring
            score = 100
            if col_res.get("Null %", 0) > 10: score -= 15
            if col_res.get("Uniqueness %", 100) < 50: score -= 15
            if col_res.get("Outlier %", 0) > 5: score -= 15
            if col_res["Is_ID"]: score += 5  # IDs are good structure-wise
            col_res["DQ_Score"] = min(max(score, 0), 100)

            results.append(col_res)

        profile_df = pd.DataFrame(results)
        dataset_score = profile_df["DQ_Score"].mean()

        # Extra: Correlation
        numeric_cols = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_cols.corr().to_dict() if not numeric_cols.empty else {}

        # Prepare JSON for LLM reasoning
        context_json = profile_df.to_dict(orient="records")

        prompt = f"""
        You are a Data Quality Analyst.
        Here are profiling results (per column stats, anomalies, scores, ID detection):

        {context_json}

        Task:
        - Summarize key dataset-level issues in plain English.
        - Highlight columns that need attention.
        - If no major issues, say so clearly.
        """

        reasoning = self.llm.predict(prompt)

        return {
            "artifact": profile_df,
            "dataset_score": dataset_score,
            "correlation_matrix": corr_matrix,
            "llm_summary": reasoning
        }
