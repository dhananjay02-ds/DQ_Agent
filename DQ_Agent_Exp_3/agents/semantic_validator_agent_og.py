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
    Semantic + Logical Validation Agent (improved actionable version).

    ✅ Deduplicates repetitive LLM descriptions
    ✅ Aggregates unique inconsistency reasons and example rows
    ✅ Returns meaningful summary and final artifact table
    ✅ Compatible with orchestrator’s artifact rendering
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

    # ---------------- Column selection & context ----------------
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
            if re.search(r"id|_id$|code|type|category|status|class|segment", name):
                candidates.append(c)
                continue
            if df[c].nunique(dropna=True) <= self.max_cardinality and not pd.api.types.is_numeric_dtype(df[c]):
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

    # ---------------- LLM prompt helpers ----------------
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

    def _build_categorical_prompt(self, target: str, context_cols: List[str], batch_df: pd.DataFrame) -> str:
        df_copy = batch_df.copy()
        for c in df_copy.select_dtypes(include=["datetime", "datetimetz"]).columns:
            df_copy[c] = df_copy[c].astype(str)
        samples = self._json_safe(df_copy[[target] + context_cols].head(20).to_dict(orient="records"))
        prompt = f"""
You are a data validation specialist focusing on semantic consistency between columns.
Target column: "{target}"
Context columns: {context_cols}

Sample rows (up to 20):
{json.dumps(samples, indent=2)}

Tasks:
1) Describe what the target column likely represents.
2) Identify inconsistent target values given the context columns.
3) Return strict JSON:
{{
  "description": "...",
  "inconsistencies": [{{"sample_idx": 0, "reason": "..."}}],
  "verdict": "consistent | possibly inconsistent | inconsistent",
  "confidence": "High | Medium | Low"
}}
"""
        return prompt

    # ---------------- Validation Logic ----------------
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
            prompt = self._build_categorical_prompt(target, context_cols, batch)
            try:
                resp = self.llm.predict(prompt)
                parsed = self._safe_json_load(resp.strip().replace("```json", "").replace("```", ""))
            except Exception:
                parsed = None

            if parsed:
                descriptions.append(parsed.get("description"))
                verdict = parsed.get("verdict", "possibly inconsistent")
                verdicts.append((verdict, parsed.get("confidence", "Low")))
                for item in parsed.get("inconsistencies", []):
                    idx, reason = item.get("sample_idx"), item.get("reason")
                    try:
                        row_snap = batch.reset_index().loc[idx].to_dict()
                    except Exception:
                        row_snap = {}
                    inconsistent_records.append({"sample_idx_batch": idx, "sample_snapshot": row_snap, "reason": reason})
            else:
                verdicts.append(("possibly inconsistent", "Low"))

        # --- Deduplicate descriptions ---
        unique_desc = list(dict.fromkeys([d.strip() for d in descriptions if d]))
        desc_summary = " | ".join(unique_desc[:3])

        # --- Aggregate unique reasons ---
        reason_texts = [r["reason"] for r in inconsistent_records if r.get("reason")]
        unique_reasons = list(dict.fromkeys(reason_texts))[:5]
        example_rows = [r["sample_snapshot"] for r in inconsistent_records if r.get("sample_snapshot")][:3]

        stats = {"avg_mode_share": 0.0, "max_cramers_v": 0.0}
        if context_cols:
            mode_shares = []
            cramers = []
            for ctx in context_cols:
                mode_shares.append(self._conditional_mode_share(df[[ctx, target]].dropna(), ctx, target))
                cramers.append(self._cramers_v(df[ctx].fillna("__NA__"), df[target].fillna("__NA__")))
            stats["avg_mode_share"] = float(np.mean(mode_shares))
            stats["max_cramers_v"] = float(np.max(cramers))

        counts = Counter([v for v, _ in verdicts])
        aggregated_verdict = "consistent"
        aggregated_confidence = "Low"
        if counts.get("inconsistent", 0) > 0:
            aggregated_verdict = "inconsistent"
        elif counts.get("possibly inconsistent", 0) > 0:
            aggregated_verdict = "possibly inconsistent"

        discrepancy_flag = (
            aggregated_verdict in ("inconsistent", "possibly inconsistent")
            and (stats["avg_mode_share"] > 0.5 or stats["max_cramers_v"] > 0.5)
        )

        return {
            "Column": target,
            "Type": "categorical",
            "Samples_Checked": rows_checked,
            "Verdict": aggregated_verdict,
            "Confidence": aggregated_confidence,
            "Discrepancy_Flag": "Classification might be wrong" if discrepancy_flag else "",
            "Example_Reasons": unique_reasons,
            "Example_Rows": example_rows,
            "Description": desc_summary,
            "Statistical_Signals": stats,
        }

    # ---------------- Aggregation and Run ----------------
    def run(self, df, focus_columns=None, max_context=3):
        findings = []
        candidates = self._select_candidates(df, focus_columns)

        if not candidates:
            return {"artifact": pd.DataFrame(), "llm_summary": "No candidate columns found for semantic validation."}

        for col in candidates:
            ctx_cols = self._select_context_columns(df, col, max_context=max_context)
            res = self._validate_categorical_column(df, col, ctx_cols)
            findings.append(res)

        artifact = pd.DataFrame(findings)
        top_issues = artifact[artifact["Discrepancy_Flag"] != ""]

        if top_issues.empty:
            llm_summary = f"Checked {len(candidates)} columns. No major semantic discrepancies detected."
        else:
            lines = []
            for _, row in top_issues.iterrows():
                reasons = "; ".join(row.get("Example_Reasons", []))
                lines.append(f"- **{row['Column']}**: {row['Discrepancy_Flag']} | {reasons}")
            llm_summary = "Detected semantic inconsistencies:\n" + "\n".join(lines)

        return {"artifact": artifact, "llm_summary": llm_summary}
