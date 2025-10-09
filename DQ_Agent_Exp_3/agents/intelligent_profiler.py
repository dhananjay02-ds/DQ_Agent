import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from llm_utils import get_llm
import re, json
from collections import Counter
from typing import Dict, Any


class IntelligentDQAgent:
    """
    Advanced Data Quality Profiler compatible with LangChain agent system.

    Features:
    - Completeness (null + placeholder)
    - Uniqueness
    - Validity (type, regex, range, referential)
    - Consistency (format & cross-column checks)
    - Accuracy (optional reference data)
    - Timeliness (based on timestamp)
    - Outliers (Z, IQR, IsolationForest)
    - Entropy
    - Correlation / Mutual Information
    - Drift (PSI or KS-test)
    - LLM-based column definition + description + reasoning + potential issues
    """

    def __init__(self, model="gpt-4o-mini", baseline: pd.DataFrame = None):
        self.llm = get_llm(model=model)
        self.baseline = baseline

    # ---------- Helper Functions ----------
    def _placeholder_ratio(self, s):
        placeholders = ["n/a", "na", "none", "unknown", "-", "", "0"]
        s_clean = s.dropna().astype(str).str.strip().str.lower()
        return (s_clean.isin(placeholders)).mean()

    def _uniqueness(self, s):
        return s.nunique(dropna=True) / len(s) if len(s) > 0 else 0

    def _entropy(self, s):
        counts = np.array(list(Counter(s.dropna()).values()), dtype=float)
        probs = counts / counts.sum() if counts.sum() > 0 else [0]
        return -np.sum(probs * np.log(probs + 1e-9))

    def _format_consistency(self, s, pattern=None):
        if pattern is None:
            pattern = r"^[a-zA-Z0-9@.\-_\s]+$"
        s_clean = s.dropna().astype(str)
        return s_clean.apply(lambda x: bool(re.match(pattern, x))).mean() if len(s_clean) else 1.0

    def _range_conformity(self, s, minv=None, maxv=None):
        if not pd.api.types.is_numeric_dtype(s):
            return 1.0
        valid = s.dropna()
        if valid.empty:
            return 1.0
        return ((valid >= (minv if minv is not None else valid.min())) &
                (valid <= (maxv if maxv is not None else valid.max()))).mean()

    def _referential_integrity(self, s, reference=None):
        if reference is None or reference.empty:
            return np.nan
        refset = set(reference.dropna().astype(str))
        return s.dropna().astype(str).apply(lambda x: x in refset).mean()

    def _outliers(self, s):
        if not pd.api.types.is_numeric_dtype(s) or len(s.dropna()) < 10:
            return {"z": 0, "iqr": 0, "iso": 0}
        x = s.dropna()
        z_rate = (np.abs(zscore(x)) > 3).mean()
        q1, q3 = np.percentile(x, [25, 75])
        iqr_rate = ((x < (q1 - 1.5 * (q3 - q1))) | (x > (q3 + 1.5 * (q3 - q1)))).mean()
        iso = IsolationForest(random_state=42, contamination="auto")
        preds = iso.fit_predict(x.values.reshape(-1, 1))
        iso_rate = (preds == -1).mean()
        return {"z": z_rate, "iqr": iqr_rate, "iso": iso_rate}

    def _psi(self, base, current, bins=10):
        try:
            base, current = np.array(base), np.array(current)
            quantiles = np.percentile(base, np.linspace(0, 100, bins + 1))
            psi = 0
            for i in range(bins):
                e = ((base >= quantiles[i]) & (base <= quantiles[i + 1])).mean()
                a = ((current >= quantiles[i]) & (current <= quantiles[i + 1])).mean()
                if e > 0 and a > 0:
                    psi += (e - a) * np.log(e / a)
            return psi
        except Exception:
            return np.nan

    # ---------- Advanced Type Detection ----------
    def _detect_id_column(self, col_name, s):
        """
        Advanced heuristic to detect identifier / index-like columns.
        Uses name patterns, uniqueness ratio, and type checks.
        """
        col_lower = col_name.lower()
        if re.search(r"(id|code|number|_key|_no)$", col_lower):
            if s.nunique(dropna=True) >= 0.9 * len(s):  # nearly unique
                return True
        # numeric or string with all unique values
        if s.nunique(dropna=True) == len(s) and len(s) > 0:
            if pd.api.types.is_integer_dtype(s) or pd.api.types.is_string_dtype(s):
                return True
        return False

    def _detect_datetime_column(self, col_name, s):
        """
        Semantic detection for datetime-like columns.
        Combines name heuristics, parse success ratio, and epoch detection.
        """
        name_patterns = ["date", "time", "timestamp", "datetime", "created", "updated", "dob"]
        if any(p in col_name.lower() for p in name_patterns):
            return True

        s_non_null = s.dropna().astype(str)
        if len(s_non_null) == 0:
            return False
        try:
            parsed = pd.to_datetime(s_non_null, errors="coerce", infer_datetime_format=True)
            if parsed.notna().mean() > 0.7:
                return True
        except Exception:
            pass

        if pd.api.types.is_numeric_dtype(s) and len(s) > 0 and s.mean() > 10_000_000_000:
            return True

        return False

    # ---------- Robust LLM JSON parsing helper ----------
    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        """
        Try to extract JSON object from an LLM response string. Returns dict or {}.
        Attempts:
          - direct json.loads
          - strip triple backticks
          - find first {...} and parse
        """
        if not isinstance(text, str):
            return {}
        t = text.strip()
        # direct
        try:
            return json.loads(t)
        except Exception:
            pass
        # remove markdown fences
        t2 = t.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(t2)
        except Exception:
            pass
        # try to extract first {...}
        m = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", t, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # fallback empty
        return {}

    # ---------- Main Logic ----------
    def run(self, df: pd.DataFrame, reference_tables: Dict[str, pd.DataFrame] = None):
        results = []
        reference_tables = reference_tables or {}

        for col in df.columns:
            s = df[col]
            total = len(s)
            non_null = s.notna().sum()
            completeness = non_null / total if total else 0
            placeholder_ratio = self._placeholder_ratio(s)
            uniqueness = self._uniqueness(s)
            entropy_val = self._entropy(s)

            dtype_conform = 1.0
            if pd.api.types.is_datetime64_any_dtype(s):
                dtype_conform = s.notna().mean()

            fmt_consistency = self._format_consistency(s)
            range_conf = self._range_conformity(s)
            referential = np.nan

            for ref_name, ref_df in reference_tables.items():
                # reference table matching - use lowercase column names for robustness
                ref_cols = [c.lower() for c in ref_df.columns]
                if col.lower() in ref_cols or col.lower().endswith("code"):
                    referential = self._referential_integrity(s, ref_df.iloc[:, 0])
                    break

            outliers = self._outliers(s)
            drift = np.nan
            if self.baseline is not None and col in self.baseline.columns:
                drift = self._psi(self.baseline[col].dropna(), s.dropna())

            validity = np.nanmean([
                dtype_conform,
                fmt_consistency,
                range_conf,
                referential if not np.isnan(referential) else 1.0,
            ])
            consistency = fmt_consistency

            # Detect special types
            is_id = self._detect_id_column(col, s)
            is_datetime = self._detect_datetime_column(col, s)

            # ----------- Scaled DQ Score (0–100) -----------
            if is_id:
                dq_score = 100.0
                dq_grade = "Identifier"
            elif is_datetime:
                try:
                    s_dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
                except Exception:
                    s_dt = pd.Series([pd.NaT] * len(s))

                valid_ratio = s_dt.notna().mean()
                if valid_ratio == 0:
                    dq_score = 0
                    dq_grade = "Invalid Datetime"
                else:
                    today = pd.Timestamp.today()
                    max_dt = s_dt.max()
                    freshness_days = (today - max_dt).days if pd.notna(max_dt) else np.nan
                    past_range = pd.notna(s_dt.min()) and (s_dt.min() > pd.Timestamp("1900-01-01"))
                    future_range = pd.notna(max_dt) and (max_dt < (today + pd.Timedelta(days=5)))
                    nonnull_dt = s_dt.dropna()
                    in_order = False
                    if len(nonnull_dt) > 1:
                        in_order = (nonnull_dt.is_monotonic_increasing or nonnull_dt.is_monotonic_decreasing)
                    elif len(nonnull_dt) == 1:
                        in_order = True

                    datetime_quality = np.nanmean([
                        valid_ratio,
                        1 - (freshness_days / 365 if (isinstance(freshness_days, (int, float)) and freshness_days >= 0) else 1),
                        1 if past_range else 0,
                        1 if future_range else 0,
                        1 if in_order else 0.8,
                    ])
                    dq_score = max(min(datetime_quality * 100, 100), 0)

                    if dq_score >= 90:
                        dq_grade = "Fresh & Valid"
                    elif dq_score >= 75:
                        dq_grade = "Valid but Slightly Stale"
                    elif dq_score >= 60:
                        dq_grade = "Old / Semi-Valid"
                    else:
                        dq_grade = "Invalid / Corrupted"
            else:
                dq_score = (
                    0.25 * completeness +
                    0.15 * (1 - placeholder_ratio) +
                    0.15 * validity +
                    0.10 * consistency +
                    0.10 * (1 - np.mean(list(outliers.values()))) +
                    0.10 * (1 - (drift if not np.isnan(drift) else 0)) +
                    0.15 * uniqueness
                ) * 100
                dq_score = max(min(dq_score, 100), 0)

                if dq_score >= 90:
                    dq_grade = "Excellent"
                elif dq_score >= 75:
                    dq_grade = "Good"
                elif dq_score >= 60:
                    dq_grade = "Moderate"
                else:
                    dq_grade = "Poor"

            col_metrics = {
                "Column": col,
                "Type": str(s.dtype),
                "Is_ID": is_id,
                "Is_Datetime": is_datetime,
                "Completeness": round(completeness, 3),
                "Placeholder %": round(placeholder_ratio * 100, 2),
                "Uniqueness": round(uniqueness, 3),
                "Entropy": round(entropy_val, 3),
                "Format Consistency": round(fmt_consistency, 3),
                "Range Conformity": round(range_conf, 3),
                "Referential Integrity": (
                    "N/A" if np.isnan(referential) else round(referential, 3)
                ),
                "Outlier_Z %": round(outliers["z"] * 100, 2),
                "Outlier_IQR %": round(outliers["iqr"] * 100, 2),
                "Outlier_ISO %": round(outliers["iso"] * 100, 2),
                "Drift (PSI)": None if np.isnan(drift) else round(drift, 4),
                "DQ_Score": round(dq_score, 2),
                "DQ_Grade": dq_grade,
                "Missing %": round(100 * (1 - completeness), 2),
                "Duplicate %": round(100 * (1 - uniqueness), 2),
            }

            # ==== LLM Reasoning ====
            non_null_samples = s.dropna().astype(str)
            sample_count = min(5, len(non_null_samples))
            samples = non_null_samples.sample(sample_count, random_state=42).tolist() if sample_count > 0 else []

            # Updated prompt: request a definition, description, reasoning, potential_issues.
            # Explicitly instruct NOT to propose remediation (GovernanceAgent will handle that).
            prompt = f"""
You are a Data Quality Analyst and Data Dictionary expert.
Given column metadata and sample values, produce a compact JSON with:
- definition: a concise human-readable definition of the column's purpose (one sentence)
- description: short summary of what the column appears to contain
- reasoning: brief explanation supporting the definition/description
- potential_issues: a list of likely data quality issues (if any)
Do NOT suggest remediation actions — remediation and rules are the responsibility of the Governance component.

Column name: {col}
Samples: {samples}
Metrics: {json.dumps(col_metrics, indent=2)}
Column type hints: Is_ID={is_id}, Is_Datetime={is_datetime}

Respond with JSON only. Example:
{{"definition":"Customer identifier","description":"Unique ID for customers","reasoning":"Name and uniqueness patterns ...","potential_issues":["missing","format inconsistency"]}}
"""

            try:
                response = self.llm.predict(prompt)
                parsed = self._safe_parse_json(response)
            except Exception:
                parsed = {}

            # populate description/definition with safe fallbacks
            definition = parsed.get("definition") or parsed.get("description") or None
            description = parsed.get("description") or parsed.get("definition") or parsed.get("reasoning") or None
            reasoning_text = parsed.get("reasoning") or None
            potential_issues = parsed.get("potential_issues") or []

            col_metrics["Definition"] = definition
            col_metrics["Description"] = description
            col_metrics["Reasoning"] = reasoning_text
            col_metrics["Potential_Issues"] = (
                ", ".join(potential_issues) if isinstance(potential_issues, list) else str(potential_issues)
            )

            results.append(col_metrics)

        df_out = pd.DataFrame(results)

        # ====== Dataset-level summary reasoning ======
        # Provide deterministic thresholds and instruct LLM to list only Moderate/Poor
        context_json = df_out.to_dict(orient="records")
        summary_prompt = f"""
You are a Senior Data Quality Expert. Use the following DQ grade thresholds:
- Excellent: DQ_Score >= 90
- Good: 75 <= DQ_Score < 90
- Moderate: 60 <= DQ_Score < 75
- Poor: DQ_Score < 60

You will be given the full column profiling (with DQ_Grade per column).
TASK:
- List ONLY columns whose DQ_Grade is 'Moderate' or 'Poor' and why (one sentence each).
- Separately list columns detected as Identifiers (Is_ID) and Datetime (Is_Datetime).
- Summarize top 3 dataset-level issues observed (missingness, outliers, drift).
- Provide an overall dataset DQ health statement (one sentence).

Important:
- Do NOT relabel columns. Trust and use the provided DQ_Grade values (Moderate/Poor/etc).
- Do NOT provide remediation steps — only point out issues and risk.

Column profiling (JSON):
{json.dumps(context_json, indent=2)}
"""

        try:
            llm_summary = self.llm.predict(summary_prompt)
        except Exception:
            llm_summary = "LLM summary generation failed."

        return {"artifact": df_out, "llm_summary": llm_summary}



# {
#     "Column": col,  
#     # → Column name as present in the dataset.

#     "Type": str(s.dtype),  
#     # → Detected data type (e.g., int64, float64, object, datetime64).  

#     "Is_ID": is_id,  
#     # → Boolean flag indicating if the column is likely an identifier (based on name & uniqueness).  

#     "Is_Datetime": is_datetime,  
#     # → Boolean flag indicating if the column represents temporal data (date/time/timestamp).  

#     "Completeness": round(completeness, 3),  
#     # → Ratio of non-null (non-missing) values to total values in the column.  
#     # Formula: non_null_count / total_count  

#     "Placeholder %": round(placeholder_ratio * 100, 2),  
#     # → Percentage of placeholder values like "NA", "None", "-", or "0" that are not true data.  
#     # Helps detect implicit missingness beyond nulls.  

#     "Uniqueness": round(uniqueness, 3),  
#     # → Ratio of distinct (unique) non-null values to total number of records.  
#     # High uniqueness suggests identifiers; low indicates categorical or repeated data.  

#     "Entropy": round(entropy_val, 3),  
#     # → Information entropy of the column values — measures variability or randomness.  
#     # Higher entropy = more diverse/uniform distribution of values.  

#     "Format Consistency": round(fmt_consistency, 3),  
#     # → Share of values conforming to a standard character format (alphanumeric, underscore, etc.).  
#     # Detects formatting anomalies in textual fields.  

#     "Range Conformity": round(range_conf, 3),  
#     # → Fraction of numeric values lying within valid min–max bounds.  
#     # By default uses observed min/max if no explicit range is provided.  

#     "Referential Integrity": (
#         "N/A" if np.isnan(referential) else round(referential, 3)
#     ),  
#     # → Ratio of values found in a reference dataset (foreign key consistency).  
#     # "N/A" means no reference table was provided for comparison.  

#     "Outlier_Z %": round(outliers["z"] * 100, 2),  
#     # → Percentage of numeric values flagged as outliers using Z-score method (>3σ).  

#     "Outlier_IQR %": round(outliers["iqr"] * 100, 2),  
#     # → Percentage of numeric values lying outside 1.5×IQR bounds (Tukey rule).  

#     "Outlier_ISO %": round(outliers["iso"] * 100, 2),  
#     # → Fraction of samples detected as outliers by Isolation Forest (model-based anomaly detection).  

#     "Drift (PSI)": None if np.isnan(drift) else round(drift, 4),  
#     # → Population Stability Index (PSI) between baseline and current distribution.  
#     # Measures dataset drift; >0.2 typically indicates significant change.  

#     "DQ_Score": round(dq_score, 2),  
#     # → Aggregated Data Quality score (0–100) combining completeness, validity, uniqueness, etc.  
#     # Weighted composite of multiple dimensions (custom weights defined in code).  

#     "DQ_Grade": dq_grade,  
#     # → Categorical label derived from DQ_Score:  
#     # “Excellent” (≥90), “Good” (75–89), “Moderate” (60–74), “Poor” (<60).  
#     # Special cases for ID or datetime fields (e.g., “Identifier”, “Fresh & Valid”).  

#     "Missing %": round(100 * (1 - completeness), 2),  
#     # → Percentage of missing (null) entries in the column.  
#     # Completeness + Missing% ≈ 100%.  

#     "Duplicate %": round(100 * (1 - uniqueness), 2),  
#     # → Share of duplicate (non-unique) values in the column.  
#     # High duplicate% may suggest redundant or low-information feature.  
# }

