# governance.py
from llm_utils import get_llm
import pandas as pd
import numpy as np
import re
import json
from typing import Dict, Any, List, Tuple


class GovernanceAgent:
    """
    Unified GovernanceAgent (single-file):
    - deterministic heuristic rules (safe templates)
    - LLM-based rule suggestions (constrained)
    - validation / flagging of data-derived thresholds
    - deduplication (keeps highest-confidence rule)
    - professional source labels
    - stakeholder-friendly pseudo-logic (short)
    - Governance Risk Index (0-100) for prioritization
    """

    def __init__(self, eda_agent=None, model="gpt-4o-mini", max_freshness_days: int = 90):
        self.llm = get_llm(model=model)
        self.eda_agent = eda_agent
        self.max_freshness_days = max_freshness_days

    # ----------------- small helpers -----------------
    def _confidence_score(self, conf: str) -> float:
        if not conf:
            return 0.5
        conf = str(conf).strip().lower()
        return {"high": 1.0, "medium": 0.7, "low": 0.4}.get(conf, 0.5)

    def _source_priority(self, src: Any) -> int:
        """Return numeric priority for a rule source (handles all data types safely)."""
        try:
            if isinstance(src, (pd.Series, list, np.ndarray)):
                src = next((str(s) for s in src if pd.notna(s)), "")
            src_str = str(src).strip().lower()
        except Exception:
            src_str = ""
        return {"heuristic": 0, "llm": 1, "llm-data-derived": 2}.get(src_str, 3)


    def _normalize_rule_text(self, text: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9 ]", " ", str(text)).lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _professionalize_source(self, src: str) -> str:
        src = (src or "").lower()
        if src == "heuristic":
            return "Rule Engine"
        if src == "llm":
            return "AI-Inferred"
        if src == "llm-data-derived":
            return "AI-Inferred (Low Confidence)"
        return str(src).title()

    def _deduplicate_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate rules keeping the highest confidence and preferred source.

        Handles:
        - Duplicate column names
        - Duplicate indexes
        - Mixed data types (list, Series, NaN)
        - Non-string Source or Rule text
        """
        if df.empty:
            return df

        # --- Defensive cleanup ---
        df = df.copy(deep=True)

        # 1️⃣ Ensure unique index
        if not df.index.is_unique:
            df = df.reset_index(drop=True)

        # 2️⃣ Ensure unique column names
        seen, new_cols = set(), []
        for c in df.columns:
            if c not in seen:
                new_cols.append(c)
                seen.add(c)
            else:
                i = 1
                while f"{c}_{i}" in seen:
                    i += 1
                new_name = f"{c}_{i}"
                new_cols.append(new_name)
                seen.add(new_name)
        df.columns = new_cols

        # 3️⃣ Ensure Source and Confidence are valid
        df["Source"] = df.get("Source", "heuristic").astype(str).fillna("heuristic")
        df["Confidence"] = df.get("Confidence", "Medium").astype(str).fillna("Medium")

        # 4️⃣ Normalize columns for dedup key
        if "Rule" not in df.columns:
            df["Rule"] = ""
        if "Column" not in df.columns:
            df["Column"] = "Unknown"

        # 5️⃣ Build unique deduplication key
        df["_key"] = df["Column"].astype(str) + "_" + df["Rule"].apply(self._normalize_rule_text)

        # 6️⃣ Compute numeric confidence & priority safely
        df["ConfidenceScore"] = df["Confidence"].apply(lambda x: self._confidence_score(str(x)))
        df["SourcePriority"] = df["Source"].apply(lambda x: self._source_priority(x))

        # 7️⃣ Sort so that best rules appear first
        df = df.sort_values(by=["ConfidenceScore", "SourcePriority"], ascending=[False, True])

        # 8️⃣ Drop duplicates (keep highest-confidence first)
        df = df.drop_duplicates(subset="_key", keep="first")

        # 9️⃣ Drop helper columns
        df.drop(columns=["_key", "ConfidenceScore", "SourcePriority"], inplace=True, errors="ignore")

        # 10️⃣ Reset index to be fully clean
        df = df.reset_index(drop=True)

        return df

    def _generate_pseudologic(self, rule_text: str, column: str) -> str:
        """Generate a concise pseudo-logic for non-technical stakeholders (uses LLM but tolerant)."""
        prompt = f"""
Convert the following data governance rule into a single-line pseudo-logic statement a business analyst can read.
Rule: "{rule_text}"
Column: "{column}"
Example output: "if {column} < 0 or {column} > 120 -> flag record"
Limit to 12 words. Plain English, no SQL.
"""
        try:
            resp = self.llm.predict(prompt)
            line = str(resp).strip().splitlines()[0]
            # protect: if LLM returns JSON or long text just fallback to short form
            if len(line.split()) > 20 or "{" in line or "}" in line:
                return f"if {column} violates rule -> flag record"
            return line
        except Exception:
            return f"if {column} violates rule -> flag record"

    def _compute_risk_index(self, severity: str, confidence: str, dq_score: float) -> float:
        severity_weight = {"High": 1.0, "Medium": 0.6, "Low": 0.3}.get(str(severity), 0.5)
        confidence_weight = {"High": 1.0, "Medium": 0.7, "Low": 0.4}.get(str(confidence), 0.5)
        dq_factor = (100 - float(dq_score)) / 100 if (dq_score is not None and not pd.isna(dq_score)) else 0.5
        risk_index = 100 * (severity_weight * confidence_weight * dq_factor)
        return round(min(max(risk_index, 0), 100), 2)

    # ----------------- original helpers -----------------
    def _safe_sql_example(self, col: str, rule_type: str, params: Dict[str, Any] = None) -> str:
        params = params or {}
        if rule_type == "not_null":
            return f"<TABLE> WHERE {col} IS NULL -- find violating rows"
        if rule_type == "unique":
            return f"SELECT {col}, COUNT(*) FROM <TABLE> GROUP BY {col} HAVING COUNT(*) > 1"
        if rule_type == "range":
            lo = params.get("min", "MIN")
            hi = params.get("max", "MAX")
            return f"SELECT * FROM <TABLE> WHERE {col} < {lo} OR {col} > {hi}"
        if rule_type == "regex":
            regex = params.get("regex", "<pattern>")
            return f"-- SELECT * FROM <TABLE> WHERE NOT ({col} ~ '{regex}')"
        if rule_type == "fk":
            ref = params.get("reference_table", "<REF_TABLE>")
            ref_col = params.get("reference_column", "<REF_COL>")
            return f"SELECT * FROM <TABLE> t LEFT JOIN {ref} r ON t.{col} = r.{ref_col} WHERE r.{ref_col} IS NULL"
        if rule_type == "staleness":
            days = params.get("max_days", self.max_freshness_days)
            return f"SELECT * FROM <TABLE> WHERE {col} < (CURRENT_DATE - INTERVAL '{days} day')"
        return f"-- No SQL example available for rule_type={rule_type}"

    def _extract_numbers(self, text: str) -> List[float]:
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        return [float(n) for n in nums] if nums else []

    def _is_data_derived_threshold(self, proposed_vals: List[float], actual_min: float, actual_max: float) -> bool:
        if not proposed_vals:
            return False
        for v in proposed_vals:
            if pd.isna(actual_min) or pd.isna(actual_max):
                continue
            if abs(v - actual_min) < 1e-9 or abs(v - actual_max) < 1e-9:
                return True
        return False

    # ----------------- heuristic rule generation (full) -----------------
    def _generate_heuristic_rules(self, profiling_df: pd.DataFrame, original_df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        rules: List[Dict[str, Any]] = []
        for _, r in profiling_df.iterrows():
            col = r.get("Column")
            dtype = str(r.get("Type", "")).lower()
            dq_score = float(r.get("DQ_Score") or 0)
            missing_pct = float(r.get("Missing %") or 0)
            duplicate_pct = float(r.get("Duplicate %") or 0)
            outlier_iso = float(r.get("Outlier_ISO %") or 0)
            referential = r.get("Referential Integrity")
            is_id = bool(r.get("Is_ID", False))
            is_datetime = bool(r.get("Is_Datetime", False))
            desc = r.get("Description") or ""

            if is_id:
                rules.append({
                    "Column": col,
                    "RuleType": "primary_key",
                    "RuleText": f"{col} must be unique and non-null (Primary Key candidate).",
                    "Severity": "High",
                    "Confidence": "High",
                    "EnforcementSQL": self._safe_sql_example(col, "unique"),
                    "Rationale": "Identifier columns are critical for joins and deduplication.",
                    "Source": "heuristic"
                })
                rules.append({
                    "Column": col,
                    "RuleType": "stability",
                    "RuleText": f"{col} should be stable across data loads (low drift).",
                    "Severity": "Medium",
                    "Confidence": "Medium",
                    "EnforcementSQL": "-- Compare distinct counts across windows or use hash checks.",
                    "Rationale": "IDs changing across loads suggest pipeline issues or rekeying.",
                    "Source": "heuristic"
                })
                continue

            if is_datetime:
                rules.append({
                    "Column": col,
                    "RuleType": "datetime_validity",
                    "RuleText": f"{col} must contain valid timestamps (parsable to datetime).",
                    "Severity": "High" if missing_pct > 20 else "Medium",
                    "Confidence": "High",
                    "EnforcementSQL": f"SELECT * FROM <TABLE> WHERE {col} IS NULL OR NOT (/* parsable to datetime */ false)",
                    "Rationale": "Timestamps are required for temporal analysis and SLA calculations.",
                    "Source": "heuristic"
                })
                rules.append({
                    "Column": col,
                    "RuleType": "staleness",
                    "RuleText": f"{col} should be refreshed at least once every {self.max_freshness_days} days for operational recency.",
                    "Severity": "High" if dq_score < 75 else "Medium",
                    "Confidence": "Medium",
                    "EnforcementSQL": self._safe_sql_example(col, "staleness", {"max_days": self.max_freshness_days}),
                    "Rationale": "Operational systems require recent timestamps for accuracy.",
                    "Source": "heuristic"
                })
                continue

            if "object" in dtype or "category" in dtype:
                rules.append({
                    "Column": col,
                    "RuleType": "domain_constraint",
                    "RuleText": f"Values in {col} should conform to a known taxonomy or reference table when available.",
                    "Severity": "Medium",
                    "Confidence": "Medium" if referential == "N/A" else "High",
                    "EnforcementSQL": self._safe_sql_example(col, "fk", {"reference_table": "<REF_TABLE>", "reference_column": "<REF_COL>"}),
                    "Rationale": "Domain constraints reduce free-text variability and standardize analysis.",
                    "Source": "heuristic"
                })
                if duplicate_pct > 80:
                    rules.append({
                        "Column": col,
                        "RuleType": "text_standardization",
                        "RuleText": f"{col} appears highly duplicated — enforce canonicalization and mapping (trim/case/stopwords).",
                        "Severity": "Medium",
                        "Confidence": "Medium",
                        "EnforcementSQL": "-- Use normalization pipeline and compare before/after counts",
                        "Rationale": "High duplicates often indicate categorical labels but may hide inconsistent text.",
                        "Source": "heuristic"
                    })

            if "int" in dtype or "float" in dtype:
                lower_name = col.lower()
                if "age" in lower_name:
                    rules.append({
                        "Column": col,
                        "RuleType": "business_range",
                        "RuleText": f"{col} should be between 0 and 120 (business rule for age).",
                        "Severity": "High",
                        "Confidence": "High",
                        "EnforcementSQL": self._safe_sql_example(col, "range", {"min": 0, "max": 120}),
                        "Rationale": "Age outside 0–120 is likely erroneous.",
                        "Source": "heuristic"
                    })
                elif any(k in lower_name for k in ["percent", "pct", "rate"]):
                    rules.append({
                        "Column": col,
                        "RuleType": "percentage_range",
                        "RuleText": f"{col} should be between 0 and 100 (percentage).",
                        "Severity": "Medium",
                        "Confidence": "High",
                        "EnforcementSQL": self._safe_sql_example(col, "range", {"min": 0, "max": 100}),
                        "Rationale": "Percent fields are expected to lie in 0–100.",
                        "Source": "heuristic"
                    })
                elif any(k in lower_name for k in ["price", "amount", "cost", "revenue", "salary"]):
                    rules.append({
                        "Column": col,
                        "RuleType": "non_negative",
                        "RuleText": f"{col} should be non-negative (>= 0).",
                        "Severity": "Medium",
                        "Confidence": "High",
                        "EnforcementSQL": self._safe_sql_example(col, "range", {"min": 0, "max": "INF"}),
                        "Rationale": "Monetary numeric fields should not be negative unless refunds are modeled separately.",
                        "Source": "heuristic"
                    })

                if outlier_iso > 5:
                    rules.append({
                        "Column": col,
                        "RuleType": "outlier_monitor",
                        "RuleText": f"Flag top outliers in {col} for manual review (ISO detection shows {outlier_iso:.1f}%).",
                        "Severity": "Medium" if dq_score >= 60 else "High",
                        "Confidence": "Medium",
                        "EnforcementSQL": "-- Use IsolationForest or percentile-based filters in pipeline",
                        "Rationale": "High outlier fraction can affect analytics and models.",
                        "Source": "heuristic"
                    })

            if missing_pct > 10:
                rules.append({
                    "Column": col,
                    "RuleType": "missing_values",
                    "RuleText": f"{col} has missing percentage > {missing_pct:.1f}% — define imputation or rejection policy.",
                    "Severity": "High",
                    "Confidence": "High",
                    "EnforcementSQL": f"SELECT COUNT(*) FROM <TABLE> WHERE {col} IS NULL",
                    "Rationale": "High missingness undermines feature reliability.",
                    "Source": "heuristic"
                })

            if referential not in [None, "N/A"] and isinstance(referential, (float, int)) and referential < 1.0:
                miss_pct = 100 * (1 - float(referential))
                rules.append({
                    "Column": col,
                    "RuleType": "referential_integrity",
                    "RuleText": f"{col} has {miss_pct:.1f}% values not found in reference; enforce FK checks.",
                    "Severity": "High" if miss_pct > 5 else "Medium",
                    "Confidence": "High",
                    "EnforcementSQL": self._safe_sql_example(col, "fk", {"reference_table": "<REF_TABLE>", "reference_column": "<REF_COL>"}),
                    "Rationale": "Missing references break joins and analytic integrity.",
                    "Source": "heuristic"
                })
        return rules

    # ----------------- LLM rule generation (full) -----------------
    def _ask_llm_for_rules(self, profiling_df: pd.DataFrame, dataset_summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        mini = profiling_df[[
            c for c in profiling_df.columns if c in {
                "Column", "Type", "Description", "DQ_Score", "DQ_Grade", "Missing %", "Duplicate %", 
                "Outlier_ISO %", "Outlier_Z %", "Outlier_IQR %", "Is_ID", "Is_Datetime", "Referential Integrity",
                "Drift (PSI)"
            }
        ]].to_dict(orient="records")

        prompt = f"""
You are a pragmatic Data Governance Specialist. You will be given a dataset profiling summary (JSON) for each column. 
For each column, suggest up to 2 realistic, business-meaningful governance rules (validation/monitoring/remediation).
Constraints (very important):
1) DO NOT create “business rules” that simply mirror the observed data distribution (for example: "min is 3.2 so set min threshold to 3.2"). 
   If you suggest a numeric threshold, explain the business rationale (domain reason) and mark its confidence as low/medium/high.
2) Use column name and provided DESCRIPTION to infer business semantics; if description is missing, say "definition unknown" and produce conservative monitoring rules (e.g., non-null check).
3) Avoid overly-technical prescriptions — provide an SQL predicate example and a short remediation step.
4) Output strict JSON only with a top-level key "rules" that is a list of objects with keys:
   - column (string)
   - rule_text (string)
   - rule_type (monitoring/validation/transform)
   - severity (High/Medium/Low)
   - enforcement_sql (string) - a short SQL predicate or sample query (use <TABLE> placeholder)
   - rationale (string)
   - confidence (High/Medium/Low)
   - source (llm)
Example output:
{{ "rules": [
  {{
    "column": "age",
    "rule_text": "Age must be between 0 and 120 (business rule).",
    "rule_type": "validation",
    "severity": "High",
    "enforcement_sql": "SELECT * FROM <TABLE> WHERE age < 0 OR age > 120",
    "rationale": "Age out of this range is practically invalid.",
    "confidence": "High",
    "source": "llm"
  }}
]}}
Dataset-level summary: {json.dumps(dataset_summary, indent=2)}
Column profiling (trimmed): {json.dumps(mini, indent=2)}
Remember: if you propose thresholds that match observed min/max, explicitly label them 'data-derived' and set confidence to Low.
"""
        response = self.llm.predict(prompt)
        response_clean = str(response).strip().replace("```json", "").replace("```", "").strip()
        try:
            llm_json = json.loads(response_clean)
            rules = llm_json.get("rules", [])
            return rules, response
        except Exception:
            return [], response

    # ----------------- LLM rule validation & enrichment -----------------
    def _validate_and_enrich_llm_rules(self, llm_rules: List[Dict[str, Any]], profiling_df: pd.DataFrame, original_df: pd.DataFrame = None) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        col_stats: Dict[str, Dict[str, Any]] = {}
        for _, r in profiling_df.iterrows():
            col = r.get("Column")
            col_stats[col] = {
                "min": None,
                "max": None,
                "dtype": r.get("Type"),
            }
            if original_df is not None and col in original_df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(original_df[col]):
                        col_stats[col]["min"] = float(original_df[col].min(skipna=True))
                        col_stats[col]["max"] = float(original_df[col].max(skipna=True))
                except Exception:
                    col_stats[col]["min"] = col_stats[col]["max"] = None

        for rule in llm_rules:
            col = rule.get("column")
            if col not in col_stats:
                rule["confidence"] = "Low"
                rule["rationale"] = (rule.get("rationale", "") + " Column not found in profiling output; verify column name.")
                rule["source"] = "llm"
                enriched.append(rule)
                continue

            nums = self._extract_numbers(str(rule.get("rule_text", "")) + " " + str(rule.get("enforcement_sql", "")))
            is_data_derived = False
            if nums and col_stats[col].get("min") is not None:
                if self._is_data_derived_threshold(nums, col_stats[col]["min"], col_stats[col]["max"]):
                    is_data_derived = True

            if is_data_derived:
                rule["confidence"] = "Low"
                rule["rationale"] = (rule.get("rationale", "") + " NOTE: threshold appears to match observed data min/max -> data-derived; requires business validation.")
                rule["source"] = "llm-data-derived"
            else:
                rule["source"] = "llm"

            enriched.append(rule)
        return enriched

    # ----------------- main run -----------------
    def run(self, df: pd.DataFrame, context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not context or "artifact" not in context:
            return {"llm_summary": "⚠️ No profiling data provided. Governance cannot run.", "artifact": None}

        profiling_df: pd.DataFrame = context["artifact"]
        original_df = df
        # compute dataset_score if not provided
        dataset_score = context.get("dataset_score")
        if dataset_score is None and "DQ_Score" in profiling_df.columns:
            try:
                dataset_score = float(profiling_df["DQ_Score"].mean())
            except Exception:
                dataset_score = None

        heuristic_rules = self._generate_heuristic_rules(profiling_df, original_df)

        dataset_summary = {
            "dataset_score": dataset_score,
            "num_columns": len(profiling_df),
            "bad_columns": profiling_df[profiling_df.get("DQ_Score", 100) < 60]["Column"].tolist() if "DQ_Score" in profiling_df.columns else []
        }

        llm_rules_raw, llm_raw_text = self._ask_llm_for_rules(profiling_df, dataset_summary)
        llm_rules = self._validate_and_enrich_llm_rules(llm_rules_raw, profiling_df, original_df)

        # combine into DataFrame with original source tags
        combined_list = []
        for r in heuristic_rules:
            combined_list.append({
                "Column": r["Column"],
                "Rule": r["RuleText"],
                "RuleType": r.get("RuleType"),
                "Severity": r.get("Severity", "Medium"),
                "Confidence": r.get("Confidence", "Medium"),
                "EnforcementSQL": r.get("EnforcementSQL", ""),
                "Rationale": r.get("Rationale", ""),
                "Source": r.get("Source", "heuristic")
            })
        for r in llm_rules:
            combined_list.append({
                "Column": r.get("column"),
                "Rule": r.get("rule_text"),
                "RuleType": r.get("rule_type"),
                "Severity": r.get("severity", "Medium"),
                "Confidence": r.get("confidence", "Medium"),
                "EnforcementSQL": r.get("enforcement_sql", ""),
                "Rationale": r.get("rationale", ""),
                "Source": r.get("source", "llm")
            })

        governance_df = pd.DataFrame(combined_list)

        # professionalize source labels for display, but maintain raw source for priority sorting internally
        governance_df["SourceRaw"] = governance_df["Source"].fillna("heuristic")
        # deduplicate using raw source & confidence
        governance_df = self._deduplicate_rules(governance_df.rename(columns={"SourceRaw": "Source"}))
        # now convert source to professional labels
        governance_df["Source"] = governance_df["Source"].apply(self._professionalize_source)

        # add pseudo logic
        governance_df["PseudoLogic"] = governance_df.apply(
            lambda r: self._generate_pseudologic(r["Rule"], r["Column"]), axis=1
        )

        # governance risk index (uses profiling DQ_Score if available)
        dq_map = {row["Column"]: row.get("DQ_Score", np.nan) for _, row in profiling_df.iterrows()} if "DQ_Score" in profiling_df.columns else {}
        governance_df["Governance Risk Index"] = governance_df.apply(
            lambda r: self._compute_risk_index(r.get("Severity", "Medium"), r.get("Confidence", "Medium"), dq_map.get(r["Column"], np.nan)),
            axis=1
        )

        # keep columns order friendly for downstream
        cols_out = ["Column", "Rule", "RuleType", "Severity", "Confidence", "Governance Risk Index", "PseudoLogic", "EnforcementSQL", "Rationale", "Source"]
        governance_df = governance_df.reindex(columns=[c for c in cols_out if c in governance_df.columns])

        # LLM summary prompt (concise)
        summary_prompt = f"""
You are a Chief Data Governance Officer.
Dataset DQ Score: {dataset_score}
Columns with issues: {dataset_summary['bad_columns']}

Governance sample (first 8 rules):
{governance_df.head(8).to_dict(orient='records')}

Summarize:
- Top 5 governance risks (bullet points)
- 3 highest-priority columns to monitor
- 3 recommended next actions and whether each is Automatable or Requires business validation
"""
        try:
            llm_summary = self.llm.predict(summary_prompt)
        except Exception as e:
            llm_summary = f"LLM summary generation failed: {e}"

        # dataset-level guidance
        dataset_rules = []
        if dataset_score is not None:
            if dataset_score < 60:
                dataset_rules.append("Dataset DQ score < 60: trigger full remediation workflow.")
            elif dataset_score < 75:
                dataset_rules.append("Dataset DQ score 60-75: schedule monitoring and targeted fixes.")
            else:
                dataset_rules.append("Dataset DQ score >= 75: continue monitoring critical rules.")

        return {
            "llm_summary": llm_summary,
            "artifact": governance_df,
            "dataset_rules": dataset_rules,
            "raw_llm_text": llm_raw_text
        }
