# drift_agent.py

import json
import re
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import ks_2samp, chi2_contingency
from llm_utils import get_llm  # your existing LLM helper


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index for numeric arrays.
    """
    try:
        eps = 1e-8
        # compute quantile-based breakpoints on expected
        quantiles = np.linspace(0, 1, buckets + 1)
        breaks = np.unique(np.quantile(expected, quantiles))
        if len(breaks) <= 2:
            # fallback to equal-width bins
            mn, mx = np.min(expected), np.max(expected)
            breaks = np.linspace(mn, mx, buckets + 1)

        exp_counts, _ = np.histogram(expected, bins=breaks)
        act_counts, _ = np.histogram(actual, bins=breaks)

        exp_pct = exp_counts / (exp_counts.sum() + eps)
        act_pct = act_counts / (act_counts.sum() + eps)

        # avoid zeros
        exp_pct = np.where(exp_pct == 0, eps, exp_pct)
        act_pct = np.where(act_pct == 0, eps, act_pct)

        psi_val = np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct))
        return float(psi_val)
    except Exception:
        return float("nan")


def _safe_categorical_props(arr: pd.Series) -> pd.Series:
    s = arr.fillna("__NA__").astype(str)
    return (s.value_counts(normalize=True)).sort_index()


class DriftAgent:
    """
    LLM-driven Drift Agent.

    - Uses LLM to interpret query and map natural terms to columns in df
    - Computes drift metrics (PSI, KS-test, chi-square) between baseline and target
    - Produces Plotly figures for distributions and temporal drift
    - Returns structured output for orchestrator to render
    """

    def __init__(self, model: str = "gpt-4o-mini", sample_size: int = 5000):
        self.llm = get_llm(model=model)
        self.sample_size = int(sample_size)

    # ---------------- LLM-based interpretation ----------------
    def _interpret_query(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ask the LLM to infer drift_type and the pieces needed to run it.
        Returns JSON with:
          - drift_type: "temporal" | "population" | "segment"
          - columns: list of columns (natural terms ok)
          - time_column: suggested time column or null
          - baseline_filter / target_filter: optional
          - group_by: optional grouping column (for segment)
          - prefer_type: numeric|categorical|auto
          - reasoning: human explanation
        """
        prompt = f"""
You are a data diagnostics planner. Given dataset columns and a user's request, infer what kind of drift comparison should be performed.

Dataset columns:
{list(df.columns)}

User request:
\"{query}\"

Decide:
- drift_type: one of ["temporal","population","segment"]
  - temporal: user is asking if value(s) changed over time
  - population: user is asking if distribution has shifted (baseline vs target)
  - segment: user is asking to compare distributions across groups (e.g., by region)
- columns: list of terms or column names to analyze (can be empty)
- time_column: name if the user referenced time-based slices or you detect a time-like column
- baseline_filter: optional filter expression for baseline (or null)
- target_filter: optional filter expression for target (or null)
- group_by: optional grouping column (or null)
- prefer_type: "numeric" | "categorical" | "auto"
- reasoning: brief explanation for your choice

Return strict JSON only. If unsure about filters, set them to null.
"""
        try:
            raw = self.llm.predict(prompt)
            raw = raw.strip()
            # try to extract JSON blob from the model response
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
            else:
                parsed = json.loads(raw)
        except Exception:
            # robust fallback
            parsed = {
                "drift_type": "population",
                "columns": None,
                "time_column": None,
                "baseline_filter": None,
                "target_filter": None,
                "group_by": None,
                "prefer_type": "auto",
                "reasoning": "fallback heuristics"
            }
        # normalize keys
        parsed.setdefault("drift_type", "population")
        parsed.setdefault("columns", parsed.get("columns", None))
        parsed.setdefault("time_column", parsed.get("time_column", None))
        parsed.setdefault("baseline_filter", parsed.get("baseline_filter", None))
        parsed.setdefault("target_filter", parsed.get("target_filter", None))
        parsed.setdefault("group_by", parsed.get("group_by", None))
        parsed.setdefault("prefer_type", parsed.get("prefer_type", "auto"))
        parsed.setdefault("reasoning", parsed.get("reasoning", ""))
        return parsed

    # ---------------- Column matching (fuzzy using name normalization) ----------------
    def _safe_col(self, name: Optional[str], df: pd.DataFrame) -> Optional[str]:
        if not name:
            return None
        name_clean = re.sub(r"[^0-9a-z]", "", str(name).strip().lower())
        for c in df.columns:
            c_clean = re.sub(r"[^0-9a-z]", "", c.lower())
            if name_clean == c_clean or name_clean in c_clean or c_clean in name_clean:
                return c
        return None

    def _map_columns(self, columns: Optional[List[str]], df: pd.DataFrame) -> List[str]:
        if not columns:
            return []
        mapped = []
        for name in columns:
            try:
                col = self._safe_col(name, df)
            except Exception:
                col = None
            if col:
                mapped.append(col)
        return list(dict.fromkeys(mapped))

    # ------------------ Filtering helpers ------------------
    def _apply_filter(self, df: pd.DataFrame, filter_expr: Optional[str]) -> pd.DataFrame:
        """
        Apply a simple filter expression if present.
        Uses pandas.query safely; rejects obviously unsafe patterns.
        """
        if not filter_expr:
            return df
        try:
            if re.search(r"[;|`]", filter_expr):
                return df
            return df.query(filter_expr)
        except Exception:
            return df

    # ------------------ Execution helpers ------------------
    def _sample(self, df: pd.DataFrame) -> pd.DataFrame:
        n = min(len(df), self.sample_size)
        if n <= 0:
            return df
        return df.sample(n=n, random_state=0) if len(df) > n else df

    def _compute_numeric_drift(self, baseline: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        try:
            ks_stat, ks_p = ks_2samp(baseline, target)
        except Exception:
            ks_stat, ks_p = float("nan"), float("nan")
        psi_val = _psi(baseline, target, buckets=10)
        return {"ks_stat": float(ks_stat), "ks_p": float(ks_p), "psi": float(psi_val)}

    def _compute_categorical_drift(self, baseline_series: pd.Series, target_series: pd.Series) -> Dict[str, Any]:
        b_props = _safe_categorical_props(baseline_series)
        t_props = _safe_categorical_props(target_series)
        all_idx = sorted(set(b_props.index.tolist()) | set(t_props.index.tolist()))
        b = np.array([b_props.get(i, 0.0) for i in all_idx])
        t = np.array([t_props.get(i, 0.0) for i in all_idx])
        try:
            b_counts = (baseline_series.fillna("__NA__").astype(str).value_counts()).reindex(all_idx).fillna(0).values
            t_counts = (target_series.fillna("__NA__").astype(str).value_counts()).reindex(all_idx).fillna(0).values
            contingency = np.vstack([b_counts, t_counts])
            chi2, p, _, _ = chi2_contingency(contingency, correction=False)
        except Exception:
            chi2, p = float("nan"), float("nan")
        psi_val = _psi(b, t, buckets=len(all_idx) if len(all_idx) > 1 else 2)
        return {"chi2": float(chi2), "chi2_p": float(p), "psi": float(psi_val), "categories": all_idx}

    # ------------------ Plot helpers ------------------
    def _plot_numeric(self, baseline: pd.Series, target: pd.Series, col: str) -> go.Figure:
        b = baseline.dropna()
        t = target.dropna()
        df_b = pd.DataFrame({col: b, "period": "baseline"})
        df_t = pd.DataFrame({col: t, "period": "target"})
        df_plot = pd.concat([df_b, df_t], ignore_index=True)
        fig = px.histogram(df_plot, x=col, color="period", nbins=30, barmode="overlay", opacity=0.65)
        fig.update_layout(title=f"Numeric distribution drift for '{col}'", legend_title_text="Period")
        return fig

    def _plot_categorical(self, baseline: pd.Series, target: pd.Series, col: str) -> go.Figure:
        b = baseline.fillna("__NA__").astype(str)
        t = target.fillna("__NA__").astype(str)
        b_counts = b.value_counts(normalize=True).rename("baseline_pct")
        t_counts = t.value_counts(normalize=True).rename("target_pct")
        df = pd.concat([b_counts, t_counts], axis=1).fillna(0).reset_index().rename(columns={"index": col})
        df_m = df.melt(id_vars=[col], value_vars=["baseline_pct", "target_pct"], var_name="period", value_name="pct")
        df_m["period"] = df_m["period"].map({"baseline_pct": "baseline", "target_pct": "target"})
        fig = px.bar(df_m, x=col, y="pct", color="period", barmode="group")
        fig.update_layout(title=f"Categorical distribution drift for '{col}'", yaxis_tickformat=".0%")
        return fig

    # ------------------ Subroutines for drift types ------------------
    def _run_population_drift(self, df: pd.DataFrame, mapped_cols: List[str],
                              baseline_df: pd.DataFrame, target_df: pd.DataFrame,
                              prefer_type: str) -> Tuple[List[Dict[str, Any]], Dict[str, go.Figure]]:
        drift_rows = []
        figs = {}
        for col in mapped_cols:
            try:
                series_b = baseline_df[col]
                series_t = target_df[col]
            except Exception:
                continue

            is_numeric = pd.api.types.is_numeric_dtype(df[col]) or (prefer_type == "numeric")
            if prefer_type == "categorical":
                is_numeric = False

            if is_numeric:
                metrics = self._compute_numeric_drift(series_b.dropna().astype(float).values,
                                                      series_t.dropna().astype(float).values)
                fig = self._plot_numeric(series_b, series_t, col)
                drift_rows.append({
                    "column": col,
                    "type": "numeric",
                    "ks_stat": metrics.get("ks_stat"),
                    "ks_p": metrics.get("ks_p"),
                    "psi": metrics.get("psi"),
                    "baseline_n": int(series_b.dropna().shape[0]),
                    "target_n": int(series_t.dropna().shape[0]),
                })
                figs[col] = fig
            else:
                metrics = self._compute_categorical_drift(series_b, series_t)
                fig = self._plot_categorical(series_b, series_t, col)
                drift_rows.append({
                    "column": col,
                    "type": "categorical",
                    "chi2": metrics.get("chi2"),
                    "chi2_p": metrics.get("chi2_p"),
                    "psi": metrics.get("psi"),
                    "baseline_n": int(series_b.shape[0]),
                    "target_n": int(series_t.shape[0]),
                })
                figs[col] = fig
        return drift_rows, figs

    def _run_temporal_drift(self, df: pd.DataFrame, mapped_cols: List[str],
                            time_col: str) -> Tuple[List[Dict[str, Any]], Dict[str, go.Figure]]:
        """
        Compute drift across consecutive time buckets (monthly by default).
        For each mapped column, compute PSI between consecutive months and return a line chart of PSI.
        """
        drift_rows = []
        figs = {}
        try:
            tmp = df[[time_col] + mapped_cols].copy()
            tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
            tmp = tmp.dropna(subset=[time_col])
            if tmp.empty:
                return drift_rows, figs
            # monthly buckets (string)
            tmp["time_bucket"] = tmp[time_col].dt.to_period("M").astype(str)
            periods = sorted(tmp["time_bucket"].unique())
            if len(periods) < 2:
                # fallback: use simple 70/30 split
                baseline_df = tmp.sample(frac=0.7, random_state=0)
                target_df = tmp.drop(baseline_df.index)
                return self._run_population_drift(tmp, mapped_cols, baseline_df, target_df, prefer_type="auto")
            for col in mapped_cols:
                psi_vals = []
                for i in range(1, len(periods)):
                    prev = tmp[tmp["time_bucket"] == periods[i - 1]][col].dropna()
                    cur = tmp[tmp["time_bucket"] == periods[i]][col].dropna()
                    if len(prev) == 0 or len(cur) == 0:
                        psi = float("nan")
                    else:
                        # numeric or categorical handled in _compute_* functions
                        try:
                            if pd.api.types.is_numeric_dtype(tmp[col]):
                                psi = _psi(prev.astype(float).values, cur.astype(float).values)
                            else:
                                b_props = _safe_categorical_props(prev)
                                t_props = _safe_categorical_props(cur)
                                all_idx = sorted(set(b_props.index.tolist()) | set(t_props.index.tolist()))
                                b = np.array([b_props.get(i, 0.0) for i in all_idx])
                                t = np.array([t_props.get(i, 0.0) for i in all_idx])
                                psi = _psi(b, t, buckets=len(all_idx) if len(all_idx) > 1 else 2)
                        except Exception:
                            psi = float("nan")
                    psi_vals.append({"period": periods[i], "psi": float(psi)})
                if len([p for p in psi_vals if not np.isnan(p["psi"])]) == 0:
                    continue
                psi_df = pd.DataFrame(psi_vals)
                fig = px.line(psi_df, x="period", y="psi", title=f"PSI over time for '{col}'")
                fig.update_layout(xaxis_title="Period (YYYY-MM)", yaxis_title="PSI")
                figs[col] = fig
                drift_rows.append({
                    "column": col,
                    "type": "temporal",
                    "metric": "psi_over_time",
                    "avg_psi": float(np.nanmean([p["psi"] for p in psi_vals])),
                    "max_psi": float(np.nanmax([p["psi"] for p in psi_vals])),
                    "periods_analyzed": len(psi_vals),
                })
        except Exception:
            pass
        return drift_rows, figs

    def _run_segment_drift(self, df: pd.DataFrame, mapped_cols: List[str],
                           group_by: str) -> Tuple[List[Dict[str, Any]], Dict[str, go.Figure]]:
        """
        For each mapped column, compute drift per group vs rest (group_i vs all others).
        Returns bar charts of distributions per group and drift metrics.
        """
        drift_rows = []
        figs = {}
        try:
            groups = df[group_by].fillna("__NA__").astype(str).unique().tolist()
            # limit groups to avoid explosion
            groups = groups[:12]
            for col in mapped_cols:
                for g in groups:
                    baseline = df[df[group_by].fillna("__NA__").astype(str) == g][col].dropna()
                    target = df[df[group_by].fillna("__NA__").astype(str) != g][col].dropna()
                    if len(baseline) == 0 or len(target) == 0:
                        continue
                    if pd.api.types.is_numeric_dtype(df[col]):
                        metrics = self._compute_numeric_drift(baseline.astype(float).values, target.astype(float).values)
                        fig = self._plot_numeric(baseline, target, col)
                        key = f"{col}__{g}"
                        figs[key] = fig
                        drift_rows.append({
                            "column": col,
                            "group": g,
                            "type": "numeric_segment",
                            "ks_stat": metrics.get("ks_stat"),
                            "ks_p": metrics.get("ks_p"),
                            "psi": metrics.get("psi"),
                            "baseline_n": int(len(baseline)),
                            "target_n": int(len(target)),
                        })
                    else:
                        metrics = self._compute_categorical_drift(baseline, target)
                        fig = self._plot_categorical(baseline, target, col)
                        key = f"{col}__{g}"
                        figs[key] = fig
                        drift_rows.append({
                            "column": col,
                            "group": g,
                            "type": "categorical_segment",
                            "chi2": metrics.get("chi2"),
                            "chi2_p": metrics.get("chi2_p"),
                            "psi": metrics.get("psi"),
                            "baseline_n": int(len(baseline)),
                            "target_n": int(len(target)),
                        })
        except Exception:
            pass
        return drift_rows, figs

    # ------------------ Main run ------------------
    def run(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """
        Run drift analysis.
        Returns:
          {
            "artifact": plotly Figure OR dict of figures,
            "drift_table": pd.DataFrame with drift metrics per column,
            "summary": str,
            "suggested_next_tool": None
          }
        """
        if df is None or df.empty:
            return {"artifact": None, "drift_table": pd.DataFrame(), "summary": "No data available", "suggested_next_tool": None}

        plan = self._interpret_query(query, df)

        # map columns
        requested_cols = plan.get("columns") or []
        mapped_cols = self._map_columns(requested_cols, df) if requested_cols else []

        # If no columns suggested, try to auto-detect likely candidates
        if not mapped_cols:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in df.columns if df[c].dtype == object or df[c].nunique(dropna=True) <= 50]
            mapped_cols = (num_cols[:2] if num_cols else []) + (cat_cols[:2] if not num_cols else [])
            mapped_cols = list(dict.fromkeys(mapped_cols))[:4]

        time_col_raw = plan.get("time_column")
        time_col = self._safe_col(time_col_raw, df) or None
        baseline_filter = plan.get("baseline_filter")
        target_filter = plan.get("target_filter")
        group_by_raw = plan.get("group_by")
        group_by = self._safe_col(group_by_raw, df) or None
        prefer_type = plan.get("prefer_type", "auto")
        drift_type = plan.get("drift_type", "population")

        # prepare baseline and target (population mode)
        baseline_df = self._apply_filter(df, baseline_filter) if baseline_filter else None
        target_df = self._apply_filter(df, target_filter) if target_filter else None

        # Branch by drift type
        drift_rows = []
        figs = {}

        if drift_type == "temporal" and time_col:
            drift_rows, figs = self._run_temporal_drift(df, mapped_cols, time_col)
        elif drift_type == "segment" and group_by:
            drift_rows, figs = self._run_segment_drift(df, mapped_cols, group_by)
        else:
            # population fallback: try to create baseline/target using filters or time split or random split
            if baseline_df is None or target_df is None:
                if time_col and time_col in df.columns:
                    try:
                        tmp = df.copy()
                        tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
                        tmp = tmp.sort_values(time_col)
                        split_idx = max(1, int(len(tmp) * 0.7))
                        baseline_df = baseline_df if baseline_df is not None else tmp.iloc[:split_idx]
                        target_df = target_df if target_df is not None else tmp.iloc[split_idx:]
                    except Exception:
                        baseline_df = baseline_df if baseline_df is not None else df.sample(frac=0.5, random_state=0)
                        target_df = target_df if target_df is not None else df.drop(baseline_df.index)
                else:
                    baseline_df = baseline_df if baseline_df is not None else df.sample(frac=0.7, random_state=0)
                    target_df = target_df if target_df is not None else df.drop(baseline_df.index)

            baseline_df = self._sample(baseline_df)
            target_df = self._sample(target_df)
            drift_rows, figs = self._run_population_drift(df, mapped_cols, baseline_df, target_df, prefer_type)

        drift_table = pd.DataFrame(drift_rows)

        if drift_table.empty:
            summary = "No columns analyzed for drift (no suitable columns detected or insufficient data)."
            return {"artifact": None, "drift_table": drift_table, "summary": summary, "suggested_next_tool": None}

        # identify important flags
        important = []
        for r in drift_rows:
            flag = False
            if r.get("psi") is not None and not np.isnan(r["psi"]) and r["psi"] > 0.2:
                flag = True
            if r.get("ks_p") is not None and not np.isnan(r.get("ks_p")) and r["ks_p"] < 0.05:
                flag = True
            if r.get("chi2_p") is not None and not np.isnan(r.get("chi2_p")) and r["chi2_p"] < 0.05:
                flag = True
            if flag:
                important.append(r.get("column") or f"{r.get('column')}:{r.get('group', '')}")

        summary_lines = [
            f"Analyzed {len(drift_rows)} drift records. Potential drift flagged for: {', '.join(important) if important else 'None'}."
        ]
        summary_lines.append(f"Plan reasoning: {plan.get('reasoning', '')}")
        summary = " ".join(summary_lines)

        # assemble artifact
        if len(figs) == 0:
            artifact = None
        elif len(figs) == 1:
            artifact = list(figs.values())[0]
        else:
            artifact = figs  # dict of figures; orchestrator.render_agent_output should handle this

        return {"artifact": artifact, "table": drift_table, "summary": summary}
