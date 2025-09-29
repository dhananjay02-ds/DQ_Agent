from llm_utils import get_llm
import pandas as pd


class InsightsAgent:
    def __init__(self, model="gpt-4o-mini"):
        self.llm = get_llm(model=model)

    def run(self, profiling_output: dict):
        profile_df = profiling_output["artifact"]
        correlation_matrix = profiling_output.get("correlation_matrix", {})

        # Column-wise reasoning
        column_insights = []
        for record in profile_df.to_dict(orient="records"):
            prompt = f"""
            You are a Data Quality Analyst.
            Column stats: {record}

            Task:
            - Point out concrete issues for this column if any.
            - Base reasoning only on provided stats.
            - If no issues, explicitly say: "No major issues detected."
            """
            reasoning = self.llm.predict(prompt)
            column_insights.append({
                "Column": record["Column"],
                "Analysis": reasoning,
                "DQ_Score": record.get("DQ_Score", None)
            })

        # Dataset-level reasoning
        dataset_prompt = f"""
        Dataset-level signals:
        Correlation Matrix: {correlation_matrix}

        Task:
        - Highlight pairs of columns that are highly correlated (>0.9 abs).
        - Keep explanation short and clear.
        """
        dataset_insight = self.llm.predict(dataset_prompt)

        # Create a dataframe for clean rendering in Streamlit
        insights_df = pd.DataFrame(column_insights)

        return {
            "artifact": insights_df,
            "llm_summary": {"dataset_level": dataset_insight}
        }
