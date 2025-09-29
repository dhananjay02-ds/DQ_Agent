from llm_utils import get_llm
import pandas as pd


class GovernanceAgent:
    def __init__(self, eda_agent=None, model="gpt-4o-mini"):
        self.llm = get_llm(model=model)
        self.eda_agent = eda_agent

    def run(self, df, context=None):
        """
        context is expected to contain:
        - artifact (DataFrame with profiling + scores + Is_ID flag)
        - dataset_score
        - correlation_matrix
        - insights (column/dataset-level insights from InsightsAgent)
        """
        if not context or "artifact" not in context:
            return {"llm_summary": "⚠️ No profiling data provided. Governance cannot run."}

        profiling_df: pd.DataFrame = context["artifact"]

        # --- Column-level governance rules ---
        governance_rules = []
        for _, row in profiling_df.iterrows():
            col = row.get("Column")
            dtype = row.get("Type", "unknown")
            rules = []

            # ID Column
            if row.get("Is_ID", False):
                rules.append("Must be unique (Primary Key).")
                rules.append("Must not contain null values.")
                rules.append("Should be stable (no drift across datasets).")

            # Numeric columns
            elif "int" in dtype or "float" in dtype:
                rules.append(f"Enforce numeric range: [{row.get('Min')}, {row.get('Max')}].")
                if row.get("Outlier %", 0) > 5:
                    rules.append("Flag high-outlier records for manual review.")
                if row.get("Null %", 0) > 0:
                    rules.append("Impute or drop null values.")

            # Categorical columns
            elif "object" in dtype:
                rules.append("Restrict values to observed categories or a reference taxonomy.")
                if row.get("Null %", 0) > 0:
                    rules.append("Fill nulls with mode or 'Unknown'.")
                if row.get("Unique Count", 0) > 50:
                    rules.append("Review for free-text values; consider standardization.")

            # Datetime columns
            elif "datetime" in dtype:
                rules.append("Ensure values fall within valid date ranges.")
                if "Freshness (days)" in row and row["Freshness (days)"] > 365:
                    rules.append("Check for stale records; enforce data refresh policy.")

            # Text columns (long strings)
            if "object" in dtype and row.get("Unique Count", 0) > 1000:
                rules.append("Enforce max length for text values (e.g., 255 chars).")

            governance_rules.append({
                "Column": col,
                "Type": dtype,
                "Governance Rules": rules
            })

        # --- Dataset-level governance ---
        dataset_rules = []
        if context.get("dataset_score", 100) < 80:
            dataset_rules.append("Dataset DQ score below threshold — initiate cleansing workflow.")

        corr_matrix = context.get("correlation_matrix", {})
        high_corr = []
        for c1, vals in corr_matrix.items():
            for c2, corr in vals.items():
                if c1 != c2 and abs(corr) > 0.9:
                    high_corr.append(f"{c1} ↔ {c2} (corr={corr:.2f})")
        if high_corr:
            dataset_rules.append("High correlation detected: " + ", ".join(high_corr))

        governance_df = pd.DataFrame(governance_rules)

        # --- Governance summary via LLM ---
        prompt = f"""
        You are a Data Governance Specialist.
        Below are column-level rules and dataset-level findings:

        Column-level rules:
        {governance_rules}

        Dataset-level rules:
        {dataset_rules}

        Task:
        - Summarize governance risks in 3–5 clear bullet points.
        - Ensure rules are actionable and realistic.
        - Do not invent unseen problems.
        """

        summary = self.llm.predict(prompt)

        return {
            "llm_summary": summary,
            "artifact": governance_df,
            "dataset_rules": dataset_rules
        }
