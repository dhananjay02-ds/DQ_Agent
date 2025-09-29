import os
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from llm_utils import get_llm

from agents.profiling import ProfilingAgent
from agents.advanced_profiler import AdvancedProfilingAgent
from agents.insights import InsightsAgent
from agents.governance import GovernanceAgent
from agents.text2df import text2df_agent
from agents.eda import eda_agent
from agents.cross_relation import CrossRelationAgent
from agents.lineage_agent import LineageAgent   # ✅ NEW

os.makedirs("outputs", exist_ok=True)


class Orchestrator:
    def __init__(self, df=None, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.df = df
        self.llm = get_llm(model=model, temperature=temperature)

        # Track outputs for Streamlit or downstream chaining
        self.tool_outputs = {}

        # Agents
        self.profiling_agent = ProfilingAgent()
        self.advanced_profiling_agent = AdvancedProfilingAgent()
        self.insights_agent = InsightsAgent()
        self.governance_agent = GovernanceAgent(eda_agent)
        self.text2df_agent = text2df_agent
        self.cross_relation_agent = CrossRelationAgent()
        self.eda_agent = eda_agent
        self.lineage_agent = LineageAgent(catalog={        # ✅ NEW
            "approved_tables": ["orders", "customers", "products"],
            "deprecated_tables": ["raw_orders"],
            "sensitive_columns": ["ssn", "dob"]
        })

        # Tools (with improved descriptions for ordering guidance)
        tools = [
            Tool(
                name="profiling",
                func=self._profiling_tool,
                description=(
                    "Run dataset profiling (dashboard-based). "
                    "Use this when the user ONLY wants a profiling report. "
                    "Do NOT use this when the user is asking about data quality issues."
                )
            ),
            Tool(
                name="advanced_profiling",
                func=self._advanced_profiling_tool,
                description=(
                    "Run advanced profiling to detect nulls, outliers, correlations, and compute a dataset quality score. "
                    "MUST be run FIRST when the user asks about data quality issues."
                )
            ),
            Tool(
                name="insights",
                func=self._insights_tool,
                description=(
                    "Analyze advanced profiling results to provide human-readable insights. "
                    "Should ONLY be used AFTER advanced_profiling has been executed."
                )
            ),
            Tool(
                name="governance",
                func=self._governance_tool,
                description=(
                    "Suggest remediation steps and generate governance/business rules "
                    "to fix data quality issues. "
                    "Consumes BOTH advanced_profiling and insights results. "
                    "Run this AFTER insights when the user asks about remediation, fixes, or rules."
                )
            ),
            Tool(
                name="cross_relation",
                func=self._cross_relation_tool,
                description=(
                    "Check consistency of relationships between categorical column pairs to find mapping errors. "
                    "Often used ALONGSIDE advanced_profiling in data quality checks. "
                    "Use when relationships or mapping consistency are relevant."
                )
            ),
            Tool(
                name="text2df",
                func=self._text2df_tool,
                description=(
                    "Convert a natural language request into a SQL query on the DataFrame. "
                    "Independent of profiling and data quality checks."
                )
            ),
            Tool(
                name="eda",
                func=self._eda_tool,
                description=(
                    "Generate visual EDA plots (distribution, correlation, counts). "
                    "Independent of profiling and data quality checks."
                )
            ),
            Tool(  # ✅ NEW
                name="lineage",
                func=self._lineage_tool,
                description=(
                    "Analyze SQL queries for data lineage. "
                    "Extracts tables, joins, filters, aggregations, and outputs. "
                    "Provides explanation and governance validation."
                )
            ),
        ]

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True
        )

        self.agent_executor = initialize_agent(
            tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True, memory=memory
        )

    # ---------------- Tool Wrappers ----------------
    def _profiling_tool(self, _input: str = ""):
        res = self.profiling_agent.run(self.df)
        self.tool_outputs["Profiling"] = res
        return res

    def _advanced_profiling_tool(self, _input: str = ""):
        res = self.advanced_profiling_agent.run(self.df)
        self.tool_outputs["Advanced Profiling"] = res
        return res

    def _insights_tool(self, _input: str = ""):
        advanced = self._advanced_profiling_tool("")
        res = self.insights_agent.run(advanced)
        self.tool_outputs["Insights"] = res
        return res

    def _governance_tool(self, _input: str = ""):
        advanced = self._advanced_profiling_tool("")
        insights = self._insights_tool("")
        context = {
            "artifact": advanced.get("artifact"),
            "dataset_score": advanced.get("dataset_score"),
            "correlation_matrix": advanced.get("correlation_matrix", {}),
            "insights": insights.get("llm_summary", {})
        }
        res = self.governance_agent.run(df=self.df, context=context)
        self.tool_outputs["Governance"] = res
        return res

    def _text2df_tool(self, user_query: str = ""):
        res = self.text2df_agent.run(self.df, query=user_query)
        self.tool_outputs["Text2DF"] = res
        return res

    def _cross_relation_tool(self, _input: str = ""):
        res = self.cross_relation_agent.run(self.df)
        self.tool_outputs["CrossRelation"] = res
        return res

    def _eda_tool(self, target_col: str = None):
        res = self.eda_agent.run(self.df, target_col=target_col)
        self.tool_outputs["EDA"] = res
        return res

    def _lineage_tool(self, sql_query: str = ""):   # ✅ NEW
        res = self.lineage_agent.run(sql_query)
        self.tool_outputs["Lineage"] = res
        return res

    # ---------------- Router ----------------
    def route(self, user_query: str, callbacks=None):
        """
        Route user queries.
        - If query starts with 'Lineage Analysis:', use LineageAgent directly.
        - Otherwise, run dataset agent flow.
        """
        self.tool_outputs = {}  # reset for each run

        # --- SQL lineage detection ---
        if user_query.startswith("Lineage Analysis:"):
            sql_query = user_query.replace("Lineage Analysis:", "").strip()
            res = self._lineage_tool(sql_query)
            return {
                "llm_response": res.get("llm_summary", "Lineage analysis complete."),
                "tool_outputs": self.tool_outputs
            }

        # --- Dataset agent flow ---
        result = self.agent_executor.run(user_query, callbacks=callbacks)
        return {
            "llm_response": result if isinstance(result, str) else str(result),
            "tool_outputs": self.tool_outputs,
        }
