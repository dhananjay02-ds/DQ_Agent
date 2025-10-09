import pandas as pd
import numpy as np
import json
import re
from typing import List, Dict, Any, Optional
from collections import Counter
from llm_utils import get_llm
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore, chi2_contingency


class SemanticValidatorAgent:
    """
    Semantic + Logical Validation Agent (LLM-driven targets + deterministic checks)

    - Uses an internal LLM pass to infer target and context columns from the user query
    - Validates categorical and numeric columns (deduped descriptions, example rows)
    - Returns structured artifact (pd.DataFrame) and human-readable llm_summary
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        n_batches: int = 5,
        batch_size: int = 10,
        max_cardinality: int = 50,
        numeric_sanity: bool = True,
        sample_random_state: int = 42,
    ):
        self.llm = get_llm(model=model)
        self.n_batches = max(1, int(n_batches))
        self.batch_size = max(1, int(batch_size))
        self.max_cardinality = max_cardinality
        self.numeric_sanity = numeric_sanity
        self.sample_random_state = sample_random_state

    # ---------------- Utility / Stats ----------------
    def _is_categorical(self, s: pd.Series) -> bool:
        return s.nunique(dropna=True) <= self.max_cardinality and not pd.api.types.is_numeric_dtype(s)

    def _safe_sample(self, df: pd.DataFrame, n: int, seed: int):
        n = min(max(1, n), len(df))
        try:
            return df.sample(n, random_state=seed)
        except Exception:
            return df.sample(n)

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        try:
            conf = pd.crosstab(x.fillna("__NA__"), y.fillna("__NA__"))
            if conf.size == 0 or conf.shape[0] < 2 or conf.shape[1] < 2:
                return 0.0
            chi2 = chi2_contingency(conf, correction=False)[0]
            n = conf.sum().sum()
            phi2 = chi2 / n
            r, k = conf.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            denom = min(kcorr - 1, rcorr - 1)
            return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0
        except Exception:
            return 0.0

    def _conditional_mode_share(self, df: pd.DataFrame, key: str, val: str) -> float:
        try:
            groups = df.groupby(key)[val].agg(lambda x: x.value_counts(normalize=True).max() if len(x) else 0)
            return float(groups.mean()) if len(groups) > 0 else 0.0
        except Exception:
            return 0.0

    def _numeric_outlier_rates(self, s: pd.Series) -> Dict[str, float]:
        out = {"z": 0.0, "iqr": 0.0, "iso": 0.0}
        try:
            numeric = pd.to_numeric(s.dropna(), errors="coerce")
            if len(numeric) < 10:
                return out
            arr = numeric.values
            z_mask = np.abs(zscore(arr)) > 3
            out["z"] = float(z_mask.mean())
            q1, q3 = np.percentile(arr, [25, 75])
            iqr_mask = (arr < (q1 - 1.5 * (q3 - q1))) | (arr > (q3 + 1.5 * (q3 - q1)))
            out["iqr"] = float(iqr_mask.mean())
            iso = IsolationForest(random_state=0, contamination="auto")
            preds = iso.fit_predict(arr.reshape(-1, 1))
            out["iso"] = float((preds == -1).mean())
        except Exception:
            pass
        return out

    # ---------------- JSON Safety ----------------
    def _json_safe(self, obj: Any) -> Any:
        """Recursively convert numpy/pandas/datetime objects to JSON-safe types."""
        import numpy as np
        import pandas as pd
        if isinstance(obj, (np.generic, np.int64, np.float64)):
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._json_safe(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items()}
        else:
            return obj

    # ---------------- LLM-based interpretation ----------------
    def _safe_json_load(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None

    def _interpret_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ask the LLM which target columns and context columns to validate for this query.
        Returns:
          {
            "target_columns": [...],
            "context_columns": [...],
            "reasoning": "..."
          }
        """
        cols = list(df.columns)
        prompt = f"""
You are a data understanding assistant. Given a dataset's column names and a user's request,
decide which column(s) the user likely means and which context columns will help validate them.

Columns available: {cols}
User request: "{query}"

Return strict JSON with:
{{
  "target_columns": ["..."],         // column names or natural terms
  "context_columns": ["..."],        // optional - 2-3 helpful context columns
  "reasoning": "Short rationale for your choice"
}}
If unsure, return empty lists.
"""
        try:
            raw = self.llm.predict(prompt)
            raw = raw.strip().replace("```json", "").replace("```", "")
            parsed = self._safe_json_load(raw)
            if not parsed:
                parsed = {}
        except Exception:
            parsed = {}

        # normalize keys
        parsed.setdefault("target_columns", [])
        parsed.setdefault("context_columns", [])
        parsed.setdefault("reasoning", "")
        return parsed

    def _map_names_to_cols(self, names: List[str], df: pd.DataFrame) -> List[str]:
        """
        Fuzzy map a list of names (LLM output) to actual df columns.
        Returns matched columns in original df order without duplicates.
        """
        if not names:
            return []
        cols = list(df.columns)
        mapped = []
        for nm in names:
            if not nm:
                continue
            nm_clean = re.sub(r"[^0-9a-z]", "", str(nm).strip().lower())
            # direct exact or substring match
            for c in cols:
                c_clean = re.sub(r"[^0-9a-z]", "", c.lower())
                if nm_clean == c_clean or nm_clean in c_clean or c_clean in nm_clean:
                    mapped.append(c)
                    break
        # If nothing matched, try a best-effort fuzzy by token overlap
        if not mapped:
            for nm in names:
                nm_tokens = set(re.findall(r"[a-z0-9]+", str(nm).lower()))
                best = None
                best_score = 0
                for c in cols:
                    c_tokens = set(re.findall(r"[a-z0-9]+", c.lower()))
                    score = len(nm_tokens & c_tokens)
                    if score > best_score:
                        best_score = score
                        best = c
                if best and best not in mapped and best_score > 0:
                    mapped.append(best)
        return list(dict.fromkeys(mapped))

    # ---------------- Column selection (fallback heuristics) ----------------
    def _select_candidates(self, df: pd.DataFrame, focus_columns: Optional[List[str]] = None) -> List[str]:
        cols = df.columns.tolist()
        if focus_columns:
            selected = []
            for fc in focus_columns:
                fc_clean = fc.strip().lower().replace(" ", "").replace("_", "")
                for c in cols:
                    c_clean = c.lower().replace("_", "")
                    if fc_clean in c_clean or c_clean in fc_clean:
                        selected.append(c)
                        break
            return list(dict.fromkeys(selected))
        candidates = []
        for c in df.columns:
            name = c.lower()
            if re.search(r"id|code|type|category|status|class|segment", name):
                candidates.append(c)
                continue
            # include numeric and low-card categorical columns
            if df[c].nunique(dropna=True) <= self.max_cardinality or pd.api.types.is_numeric_dtype(df[c]):
                candidates.append(c)
        return list(dict.fromkeys(candidates))

    def _select_context_columns(self, df: pd.DataFrame, target: str, max_context: int = 3) -> List[str]:
        cand = []
        for c in df.columns:
            if c == target:
                continue
            mean_len = df[c].astype(str).dropna().map(len).mean() if len(df[c].dropna()) else 0
            score = 0
            if re.search(r"name|title|desc|label|product|fund", c.lower()):
                score += 2
            if mean_len > 5:
                score += 1
            if df[c].nunique(dropna=True) > 1:
                score += 0.5
            if score > 0:
                cand.append((c, score))
        cand = sorted(cand, key=lambda x: -x[1])
        return [c for c, _ in cand][:max_context]

    # ---------------- Main Validation Logic ----------------
    def _validate_categorical_column(self, df, target, context_cols):
        rng = np.random.RandomState(self.sample_random_state)
        inconsistent_records, descriptions, verdicts = [], [], []
        rows_checked = 0

        for b in range(self.n_batches):
            seed = int(rng.randint(0, 2**31 - 1))
            batch = self._safe_sample(df, min(self.batch_size, len(df)), seed)
            if batch.empty:
                continue
            rows_checked += len(batch)

            # guard: if context_cols not present in batch, reduce to existing
            actual_ctx = [c for c in context_cols if c in batch.columns]
            prompt = f"""
You are a data validation specialist focusing on semantic consistency between columns.
Target: "{target}"
Context: {actual_ctx}

Sample rows:
{json.dumps(self._json_safe(batch[[target] + actual_ctx].head(15).to_dict(orient="records")), indent=2)}

Tasks:
- Identify inconsistent or logically incorrect values for {target} given the context.
Return JSON:
{{
  "description": "...",
  "inconsistencies": [{{"sample_idx": 0, "reason": "..."}}],
  "verdict": "consistent | possibly inconsistent | inconsistent",
  "confidence": "High | Medium | Low"
}}
"""
            try:
                parsed = self._safe_json_load(self.llm.predict(prompt))
            except Exception:
                parsed = None

            if parsed:
                descriptions.append(parsed.get("description"))
                verdicts.append((parsed.get("verdict", "possibly inconsistent"), parsed.get("confidence", "Low")))
                for item in parsed.get("inconsistencies", []):
                    idx, reason = item.get("sample_idx"), item.get("reason")
                    try:
                        row_snap = batch.reset_index().loc[idx].to_dict()
                    except Exception:
                        row_snap = {}
                    inconsistent_records.append({"sample_snapshot": row_snap, "reason": reason})

        # Deduplicate description and aggregate reasons/examples
        desc_summary = " | ".join(list(dict.fromkeys([d.strip() for d in descriptions if d]))[:3])
        reason_texts = [r["reason"] for r in inconsistent_records if r.get("reason")]
        unique_reasons = list(dict.fromkeys(reason_texts))[:5]
        example_rows = [r["sample_snapshot"] for r in inconsistent_records if r.get("sample_snapshot")][:3]

        counts = Counter([v for v, _ in verdicts])
        verdict = "consistent"
        if counts.get("inconsistent", 0) > 0:
            verdict = "inconsistent"
        elif counts.get("possibly inconsistent", 0) > 0:
            verdict = "possibly inconsistent"

        discrepancy_flag = verdict in ("inconsistent", "possibly inconsistent")

        return {
            "Column": target,
            "Type": "categorical",
            "Samples_Checked": rows_checked,
            "Verdict": verdict,
            "Confidence": "High" if discrepancy_flag else "Medium",
            "Discrepancy_Flag": "Classification might be wrong" if discrepancy_flag else "",
            "Example_Reasons": unique_reasons,
            "Example_Rows": example_rows,
            "Description": desc_summary,
        }

    def _validate_numeric_column(self, df, target, context_cols):
        """
        LLM-driven semantic numeric validation (final version).
    
        ✔ Uses LLM to reason if negatives, zeros, or large magnitudes make sense given the column meaning
        ✔ Checks for logical consistency (e.g., ReturnRate > 1000% → invalid, FundSize = 10B → valid)
        ✔ Provides clear example rows and explanations
        ✔ No hardcoded numeric thresholds
        """
    
        # --- 1️⃣ Extract numeric data safely ---
        s = pd.to_numeric(df[target], errors="coerce")
        non_null = s.dropna()
        if len(non_null) == 0:
            return None
    
        rows_checked = len(non_null)
    
        # --- 2️⃣ Prepare semantic context for reasoning ---
        numeric_summary = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "median": float(s.median()),
            "n_unique": int(s.nunique()),
        }
    
        # Create prompt that asks the LLM to reason semantically
        prompt = f"""
    You are a semantic data validation expert.
    
    We are validating the numeric column "{target}" in a financial or tabular dataset.
    
    ### Dataset Context:
    Columns available: {list(df.columns)}
    Context columns for reasoning: {context_cols}
    
    ### Column Summary:
    {json.dumps(numeric_summary, indent=2)}
    
    ### Sample Rows (up to 15):
    {json.dumps(self._json_safe(df[[target] + context_cols].head(15).to_dict(orient="records")), indent=2)}
    
    ### Your Tasks:
    1. Analyze if the numeric values for "{target}" make **semantic sense** given their meaning and context.
       Examples:
       - Negative values in "ReturnRate" or "NetFlow" are valid (loss/outflow).
       - Negative values in "Age", "AUM", or "FundSize" are invalid.
       - "ReturnRate" > 1000% or extremely large numbers are suspicious.
       - Values consistently near 0 may indicate missing or invalid data.
    2. If issues exist, identify 1–3 representative examples with reasons.
    3. Return a short description and overall verdict.
    
    Return strict JSON only:
    {{
      "description": "short natural language explanation",
      "issues": [{{"sample_idx": 0, "reason": "..."}}],
      "verdict": "valid | suspicious | invalid"
    }}
    """
    
        parsed = None
        try:
            resp = self.llm.invoke(prompt)  # ✅ use .invoke instead of .predict
            parsed = self._safe_json_load(resp)
        except Exception:
            pass
    
        # --- 3️⃣ Parse and aggregate LLM reasoning ---
        llm_desc, llm_issues = "", []
        if parsed:
            llm_desc = parsed.get("description", "")
            llm_issues = parsed.get("issues", [])
    
        # --- 4️⃣ Extract example rows and reasons ---
        example_rows, example_reasons = [], []
        for issue in llm_issues:
            idx = issue.get("sample_idx")
            reason = issue.get("reason")
            if idx is not None and idx in df.index:
                row_snap = df.loc[idx, [target] + context_cols].to_dict()
                example_rows.append(row_snap)
            if reason:
                example_reasons.append(reason)
    
        # Deduplicate and truncate
        example_reasons = list(dict.fromkeys(example_reasons))[:5]
        example_rows = example_rows[:5]
    
        # --- 5️⃣ Final decision logic ---
        llm_verdict = parsed.get("verdict", "valid") if parsed else "valid"
        discrepancy_flag = llm_verdict in ["invalid", "suspicious"]
    
        return {
            "Column": target,
            "Type": "numeric",
            "Samples_Checked": rows_checked,
            "Verdict": llm_verdict,
            "Confidence": "High" if discrepancy_flag else "High",
            "Discrepancy_Flag": "Logical or semantic inconsistency" if discrepancy_flag else "",
            "Example_Reasons": example_reasons,
            "Example_Rows": example_rows,
            "Description": llm_desc or "No semantic inconsistencies detected.",
        }


    # ---------------- Run (LLM-driven target selection + validation) ----------------
    def run(self, df: pd.DataFrame, query: Optional[str] = None, focus_columns: Optional[List[str]] = None, max_context: int = 3, include_numeric: bool = True):
        """
        If 'query' is provided, the agent uses its internal LLM to pick target_columns and context_columns.
        If LLM fails to map targets, fallback to heuristics or 'focus_columns' if provided.
        Returns: {"artifact": pd.DataFrame, "llm_summary": str}
        """
        findings = []

        # 1) Interpret query via LLM to identify target + context (agent-level responsibility)
        target_cols = []
        context_cols_global: List[str] = []
        if query:
            plan = self._interpret_query(query, df)
            tgt_names = plan.get("target_columns", []) or []
            ctx_names = plan.get("context_columns", []) or []
            # map LLM names to actual df columns
            target_cols = self._map_names_to_cols(tgt_names, df)
            context_cols_global = self._map_names_to_cols(ctx_names, df)

        # 2) If no targets from LLM, respect explicit focus_columns if provided, else fallback to heuristics
        if not target_cols:
            if focus_columns:
                target_cols = self._map_names_to_cols(focus_columns, df)
            else:
                target_cols = self._select_candidates(df, None)

        if not target_cols:
            return {"artifact": pd.DataFrame(), "llm_summary": "No candidate columns found for semantic validation."}

        # 3) Run validation for each target
        for col in target_cols:
            # choose context: prefer LLM-chosen context then heuristic
            ctx_cols = context_cols_global or self._select_context_columns(df, col, max_context=max_context)
            if pd.api.types.is_numeric_dtype(df[col]) and include_numeric:
                res = self._validate_numeric_column(df, col, ctx_cols)
            else:
                res = self._validate_categorical_column(df, col, ctx_cols)
            if res:
                findings.append(res)

        artifact = pd.DataFrame(findings)

        # 4) Build human-readable summary (including reasons and examples)
        top_issues = artifact[artifact["Discrepancy_Flag"] != ""]

        if top_issues.empty:
            llm_summary = f"Checked {len(target_cols)} target columns. No major semantic or logical issues detected."
        else:
            lines = []
            for _, row in top_issues.iterrows():
                reasons = "; ".join(row.get("Example_Reasons", []))
                # show 1 example row as JSON snippet if exists
                ex_rows = row.get("Example_Rows", [])
                ex_snippet = ""
                if ex_rows and len(ex_rows) > 0:
                    try:
                        ex_snippet = json.dumps(ex_rows[0], default=str)
                    except Exception:
                        ex_snippet = str(ex_rows[0])
                lines.append(f"- **{row['Column']}**: {row['Discrepancy_Flag']} | {reasons} | example: {ex_snippet}")
            llm_summary = "Detected issues:\n" + "\n".join(lines)

        return {"artifact": artifact, "llm_summary": llm_summary}
