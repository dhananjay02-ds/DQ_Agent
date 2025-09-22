# agents/profiling.py
# agents/profiling.py
import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)

class ProfilingAgent:
    def __init__(self):
        pass

    def run(self, df: pd.DataFrame, query: str = None, target_col: str = None):
        """
        Returns profiling summary as both a DataFrame and a text preview.
        """
        summary = []
        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            pct_missing = float(s.isna().mean() * 100)
            unique_vals = int(s.nunique(dropna=True))
            min_val = s.min() if pd.api.types.is_numeric_dtype(s) else None
            max_val = s.max() if pd.api.types.is_numeric_dtype(s) else None
            mean_val = s.mean() if pd.api.types.is_numeric_dtype(s) else None
            top_vals = ", ".join([str(v) for v in s.value_counts(dropna=True).head(3).index.tolist()])

            summary.append({
                "Column Name": col,
                "Type": dtype,
                "% Missing": f"{pct_missing:.2f}%",
                "Unique Values": unique_vals,
                "Min": min_val,
                "Max": max_val,
                "Mean": mean_val,
                "Top Value(s)": top_vals
            })

        summary_df = pd.DataFrame(summary)
        out_path = os.path.join("outputs", "profiling_summary.csv")
        summary_df.to_csv(out_path, index=False)

        # Short summary string
        top_missing = summary_df.sort_values("% Missing", ascending=False).head(3)
        text_summary = "Top missing columns: " + "; ".join(
            [f"{r['Column Name']} ({r['% Missing']})" for _, r in top_missing.iterrows()]
        )

        return {
            "summary_path": out_path,
            "summary_df": summary_df,
            "text_summary": text_summary
        }

profiling_agent = ProfilingAgent()
