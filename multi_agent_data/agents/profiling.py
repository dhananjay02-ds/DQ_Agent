import os
import pandas as pd
from ydata_profiling import ProfileReport
import sweetviz as sv

class ProfilingAgent:
    def __init__(self):
        os.makedirs("dq_reports", exist_ok=True)

    def run(self, df: pd.DataFrame):
        # --- YData Profiling ---
        ydata_report = ProfileReport(df, title="Dataset Profiling Report", explorative=True)
        ydata_path = os.path.abspath("dq_reports/ydata_profile.html")   # absolute path
        ydata_report.to_file(ydata_path)

        # --- Sweetviz Profiling (commented for now) ---
        # sweetviz_report = sv.analyze(df)
        # sweetviz_path = "dq_reports/sweetviz_report.html"
        # sweetviz_report.show_html(sweetviz_path)

        # --- Basic Stats Table ---
        summary = []
        for col in df.columns:
            s = df[col]
            summary.append({
                "Column": col,
                "Type": str(s.dtype),
                "Missing %": round(100 * s.isna().mean(), 2),
                "Unique Values": s.nunique(),
            })
        summary_df = pd.DataFrame(summary)

        return {
            "llm_summary": "✅ Profiling complete. Dashboards saved to file.",
            "artifact": summary_df,
            "dashboards": {"ydata": ydata_path}  # File-based, not inline
        }

profiling_agent = ProfilingAgent()
