import pandas as pd
import numpy as np
from dowhy import CausalModel
from llm_utils import get_llm
import re, json
from typing import Dict, Any, List


class CausalAgent:
    """
    Causal Inference & Impact Analysis Agent (Finance-Ready)

    Features:
    - Identify causal effect of an intervention (e.g., campaign, rate change) on financial outcomes.
    - Handles observational data with confounders (backdoor adjustment).
    - Automatically detects treatment/outcome candidates.
    - Computes ATE, confidence intervals, and significance.
    - Handles datetime/object dtypes safely.
    - Provides LLM-generated explanation for business stakeholders.
    - Returns structured artifact + natural-language summary.
    """

    def __init__(self, model="gpt-4o-mini"):
        self.llm = get_llm(model=model)

    # ---------- Helper Functions ----------
    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            return {}
        t = text.strip()
        try:
            return json.loads(t)
        except Exception:
            pass
        t2 = t.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(t2)
        except Exception:
            pass
        m = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}

    def _extract_vars_from_query(self, query: str, df_cols: List[str]) -> Dict[str, Any]:
        """
        LLM-aided parsing of user query into treatment, outcome, and confounders.
        Example:
            "impact of campaign_flag on aum_change controlling for risk_score, age"
        """
        prompt = f"""
        You are a data scientist. Given a dataset with columns {df_cols},
        extract treatment, outcome, and confounders from this user request.

        Request: "{query}"

        Respond as JSON:
        {{
          "treatment": "<treatment_column>",
          "outcome": "<outcome_column>",
          "confounders": ["<list>", "<of>", "<columns>"]
        }}
        """

        try:
            parsed = self._safe_parse_json(self.llm.predict(prompt))
        except Exception:
            parsed = {}

        treatment = parsed.get("treatment")
        outcome = parsed.get("outcome")
        confounders = parsed.get("confounders", [])
        return {
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
        }

    # ---------- Main Logic ----------
    def run(self, df: pd.DataFrame, query: str = None, treatment=None, outcome=None, confounders=None):
        """
        Runs causal inference given user query or direct variable specification.
        """

        # Infer variables if not provided
        if query:
            parsed = self._extract_vars_from_query(query, df.columns.tolist())
            treatment = treatment or parsed["treatment"]
            outcome = outcome or parsed["outcome"]
            confounders = confounders or parsed["confounders"]

        if not treatment or not outcome:
            return {
                "llm_summary": "[CausalAgent] Could not infer treatment or outcome from query.",
                "artifact": None,
            }

        if treatment not in df.columns or outcome not in df.columns:
            # Auto-detect AUM change if needed
            if "aum_before" in df.columns and "aum_after" in df.columns:
                df["aum_change"] = df["aum_after"] - df["aum_before"]
                outcome = "aum_change"
            else:
                return {
                    "llm_summary": f"[CausalAgent] Could not locate treatment '{treatment}' or outcome '{outcome}' in data.",
                    "artifact": None,
                }

        confounders = [c for c in confounders if c in df.columns] or []

        df_clean = df[[treatment, outcome] + confounders].copy().dropna()
        if df_clean.empty:
            return {
                "llm_summary": "[CausalAgent] No valid rows after dropping NA. Check column selection.",
                "artifact": None,
            }

        # ---------- Data Type Handling ----------
        dropped_cols = []
        converted_cols = []

        # Handle datetime columns (convert or drop)
        datetime_cols = [c for c in df_clean.columns if np.issubdtype(df_clean[c].dtype, np.datetime64)]
        for col in datetime_cols:
            try:
                df_clean[col] = (df_clean[col] - df_clean[col].min()).dt.days
                converted_cols.append(col)
            except Exception:
                df_clean.drop(columns=[col], inplace=True)
                dropped_cols.append(col)

        # Convert objects/categoricals to numeric codes
        for c in df_clean.columns:
            if df_clean[c].dtype == "object":
                df_clean[c] = df_clean[c].astype("category").cat.codes
                converted_cols.append(c)

        # Keep only numeric
        df_clean = df_clean.select_dtypes(include=[np.number])
        if df_clean.empty:
            return {
                "llm_summary": "[CausalAgent] No usable numeric columns after cleaning and type alignment.",
                "artifact": None,
            }

        # ---------- Run Causal Model ----------
        try:
            model = CausalModel(
                data=df_clean,
                treatment=treatment,
                outcome=outcome,
                common_causes=[c for c in confounders if c in df_clean.columns],
            )
            identified = model.identify_effect()
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.propensity_score_matching"
            )

            ate = estimate.value
            ci = estimate.get_confidence_intervals()
            p_val = getattr(getattr(estimate, "estimator_object", None), "_observed_estimator", None)
            if hasattr(p_val, "p_value"):
                p_val = p_val.p_value
            else:
                p_val = None

        except Exception as e:
            return {"llm_summary": f"[CausalAgent] Error running causal model: {e}", "artifact": None}

        # ---------- Narrative Generation ----------
        explain_prompt = f"""
        You are a financial data scientist. Interpret this causal analysis result in business terms.
        Treatment: {treatment}
        Outcome: {outcome}
        Confounders: {confounders}
        Average Treatment Effect (ATE): {ate}
        Confidence Interval: {ci}
        P-value: {p_val}
        Columns converted: {converted_cols}
        Columns dropped: {dropped_cols}

        Explain this result for executives:
        - Is the effect statistically significant?
        - What does it mean for AUM or campaign strategy?
        - Which client segment might be most impacted?
        - Mention if any columns were dropped or converted.
        - Conclude with a business recommendation.
        """

        try:
            llm_summary = self.llm.predict(explain_prompt)
        except Exception:
            llm_summary = f"Causal effect of {treatment} on {outcome}: {ate} (CI={ci})."

        # ---------- Artifact Construction ----------
        artifact = pd.DataFrame([
            {
                "Treatment": treatment,
                "Outcome": outcome,
                "Confounders": ", ".join(confounders),
                "ATE": round(ate, 3) if isinstance(ate, (float, int)) else ate,
                "Confidence Interval": ci,
                "P-value": p_val,
                "Rows Used": len(df_clean),
                "Converted Columns": ", ".join(converted_cols),
                "Dropped Columns": ", ".join(dropped_cols),
            }
        ])

        return {"artifact": artifact, "llm_summary": llm_summary}
