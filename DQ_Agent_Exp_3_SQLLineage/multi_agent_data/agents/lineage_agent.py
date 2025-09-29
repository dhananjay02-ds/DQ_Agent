import sqlglot
import sqlglot.expressions as exp

class LineageAgent:
    """
    Lineage Agent for SQL queries.
    Extracts sources, joins, filters, aggregations, outputs,
    and explains lineage in both structured and natural language formats.
    """

    def __init__(self, catalog: dict = None):
        """
        Args:
            catalog (dict, optional): Governance catalog (unused for now)
        """
        self.catalog = catalog or {}

    def _extract_lineage(self, query: str):
        parsed = sqlglot.parse_one(query)
        lineage = {
            "sources": [],
            "joins": [],
            "filters": [],
            "aggregations": [],
            "outputs": []
        }

        # Sources
        for t in parsed.find_all(exp.Table):
            lineage["sources"].append(t.name)

        # Joins
        for j in parsed.find_all(exp.Join):
            cond = j.args.get("on")
            if cond:
                lineage["joins"].append(cond.sql())

        # Filters
        for w in parsed.find_all(exp.Where):
            lineage["filters"].append(w.this.sql())

        # Aggregations
        for agg in parsed.find_all(exp.AggFunc):
            lineage["aggregations"].append(agg.sql())

        # Outputs
        for sel in parsed.find_all(exp.Alias):
            lineage["outputs"].append(sel.sql())

        return lineage

    def _explain_lineage_structured(self, lineage: dict) -> str:
        """
        Returns a markdown-friendly structured explanation.
        """
        parts = ["### 🧭 SQL Lineage Summary"]

        if lineage["sources"]:
            parts.append("**Sources:**")
            parts.extend([f"- {src}" for src in lineage["sources"]])

        if lineage["joins"]:
            parts.append("\n**Joins:**")
            parts.extend([f"- {j}" for j in lineage["joins"]])

        if lineage["filters"]:
            parts.append("\n**Filters:**")
            parts.extend([f"- {f}" for f in lineage["filters"]])

        if lineage["aggregations"]:
            parts.append("\n**Aggregations:**")
            parts.extend([f"- {agg}" for agg in lineage["aggregations"]])

        if lineage["outputs"]:
            parts.append("\n**Outputs:**")
            parts.extend([f"- {o}" for o in lineage["outputs"]])

        return "\n".join(parts)

    def _explain_lineage_natural(self, lineage: dict) -> str:
        """
        Returns a natural language summary.
        """
        explanation = []

        if lineage["sources"]:
            explanation.append(f"The query reads data from {', '.join(lineage['sources'])}.")
        if lineage["joins"]:
            explanation.append(f"It applies joins based on conditions like {', '.join(lineage['joins'])}.")
        if lineage["filters"]:
            explanation.append(f"Filtering is applied on conditions such as {', '.join(lineage['filters'])}.")
        if lineage["aggregations"]:
            explanation.append(f"It performs aggregations including {', '.join(lineage['aggregations'])}.")
        if lineage["outputs"]:
            explanation.append(f"The final output includes fields such as {', '.join(lineage['outputs'])}.")

        return " ".join(explanation)

    def run(self, sql_query: str, context: dict = None):
        """
        Args:
            sql_query (str): SQL query string
            context (dict, optional): extra context (unused for now)

        Returns:
            dict: {
                "artifact": lineage JSON,
                "llm_summary": structured + natural explanation
            }
        """
        lineage = self._extract_lineage(sql_query)

        structured = self._explain_lineage_structured(lineage)
        natural = self._explain_lineage_natural(lineage)

        summary = f"{structured}\n\n---\n\n### 📝 Explanation\n{natural}"

        return {
            "artifact": lineage,
            "llm_summary": summary
        }
