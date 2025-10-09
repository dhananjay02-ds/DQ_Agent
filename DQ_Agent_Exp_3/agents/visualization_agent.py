import pandas as pd
import plotly.express as px
import json
from typing import Dict, Any, Optional
from llm_utils import get_llm


class LLMVisualizationAgent:
    """
    Advanced Visualization Agent (Single-column aware)

    - Understands natural language queries (e.g., "distribution of AUM")
    - Handles single-column and multi-column plots
    - Uses LLM for intent + schema mapping
    - Executes deterministically with Plotly

    Supports:
        trend, comparison, distribution, correlation, composition (pie)
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = get_llm(model=model)

    # ---------------------- Step 1: Query Interpretation ----------------------
    def _interpret_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        prompt = f"""
You are a data visualization planner.
Given the dataset columns and user query, identify what kind of plot should be made.

Dataset columns:
{list(df.columns)}

User query:
"{query}"

Choose:
- chart_type: one of [line, bar, scatter, box, pie, histogram]
- x: name of main column (x-axis)
- y: name of second column if required (else null)
- hue: optional grouping or color column
- reasoning: short sentence explaining your decision

Return strict JSON:
{{
  "chart_type": "line | bar | scatter | box | pie | histogram",
  "x": "<column_name>",
  "y": "<column_name or null>",
  "hue": "<column_name or null>",
  "reasoning": "..."
}}
"""
        try:
            raw = self.llm.predict(prompt)
            raw = raw.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(raw)
        except Exception:
            parsed = {"chart_type": "bar", "x": None, "y": None, "hue": None, "reasoning": "fallback parsing"}

        return parsed

    # ---------------------- Step 2: Fuzzy Column Matching ----------------------
    def _safe_col(self, name: Optional[str], df: pd.DataFrame) -> Optional[str]:
        if not name:
            return None
        name_clean = name.strip().lower().replace(" ", "").replace("_", "")
        for c in df.columns:
            c_clean = c.lower().replace("_", "")
            if name_clean in c_clean or c_clean in name_clean:
                return c
        return None

    # ---------------------- Step 3: Visualization Execution ----------------------
    def run(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        plan = self._interpret_query(query, df)
        chart_type = plan.get("chart_type", "bar").lower()
        x_col = self._safe_col(plan.get("x"), df)
        y_col = self._safe_col(plan.get("y"), df)
        hue_col = self._safe_col(plan.get("hue"), df)

        # Handle single-column intent
        single_col_mode = False
        if y_col is None or y_col == "":
            single_col_mode = True

        try:
            if chart_type == "histogram" or single_col_mode:
                fig = px.histogram(df, x=x_col, color=hue_col)
                chart_type = "histogram"
            elif chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, color=hue_col)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, trendline="ols")
            elif chart_type == "box":
                fig = px.box(df, x=x_col, y=y_col, color=hue_col)
            elif chart_type == "pie":
                if y_col and y_col in df.columns:
                    fig = px.pie(df, names=x_col, values=y_col, color=hue_col)
                else:
                    agg_df = df[x_col].value_counts().reset_index()
                    agg_df.columns = [x_col, "Count"]
                    fig = px.pie(agg_df, names=x_col, values="Count")
            else:  # default bar
                fig = px.bar(df, x=x_col, y=y_col if y_col else None, color=hue_col)
        except Exception as e:
            return {
                "artifact": None,
                "summary": f"Plot generation failed: {e}",
                "suggested_next_tool": None,
            }

        # Summarize outcome
        if single_col_mode:
            summary = (
                f"Generated a {chart_type} plot showing distribution of '{x_col}'. "
                f"Reasoning: {plan.get('reasoning', '')}"
            )
        else:
            summary = (
                f"Generated a {chart_type} plot with x='{x_col}', y='{y_col}', hue='{hue_col}'. "
                f"Reasoning: {plan.get('reasoning', '')}"
            )

        return {
            "artifact": fig,
            "summary": summary,
            "suggested_next_tool": None,
        }
