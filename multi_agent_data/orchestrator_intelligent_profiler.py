import os
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from llm_utils import get_llm

from agents.profiling import ProfilingAgent
from agents.intelligent_profiler import IntelligentDQAgent  # new agent file
from agents.governance import GovernanceAgent
from agents.text2df import text2df_agent
from agents.eda import eda_agent
from agents.cross_relation import CrossRelationAgent

os.makedirs("outputs", exist_ok=True)


class Orchestrator:
    def __init__(self, df, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.df = df
        self.llm = get_llm(model=model, temperature=temperature)

        # Track outputs for Streamlit
        self.tool_outputs = {}

        # Agents
        self.profiling_agent = ProfilingAgent()                # For user-requested profiling / EDA
        self.intelligent_dq_agent = IntelligentDQAgent()      # For DQ assessment (DQ score + issues)
        self.governance_agent = GovernanceAgent(eda_agent)
        self.text2df_agent = text2df_agent
        self.cross_relation_agent = CrossRelationAgent()

        # Tools
        tools = [
            Tool(
                name="profiling",
                func=self._profiling_tool,
                description="Run dataset profiling (schema and summary-based). Use this when the user asks to 'profile' or 'see profiling/EDA'."
            ),
            Tool(
                name="dq_assessment",                                   # <-- renamed to avoid confusion
                func=self._dq_assessment_tool,
                description=(
                    "Perform data quality assessment (DQ Score + issue detection). "
                    "Computes dataset- and column-level Data Quality Score, detects issues "
                    "such as missing values/placeholders, format/type violations, outliers, "
                    "referential integrity failures, and distributional drift. "
                    "Also provides LLM-driven description, reasoning and remediation suggestions. "
                    "Use this tool when the user intends to *check data quality or find issues*. "
                    "For general profiling/EDA use the 'profiling' tool."
                )
            ),
            Tool(
                name="governance",
                func=self._governance_tool,
                description="Consume DQ assessment results to suggest governance rules, issues, and remediations."
            ),
            Tool(
                name="text2df",
                func=self._text2df_tool,
                description="Convert natural language to SQL and query the dataframe."
            ),
            Tool(
                name="cross_relation",
                func=self._cross_relation_tool,
                description="Check consistency of relationships between categorical column pairs."
            )
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

    def _dq_assessment_tool(self, _input: str = ""):
        """
        The intended tool for checking issues and DQ score.
        This replaces the previous 'advanced_profiling' + 'insights' redundancy.
        """
        res = self.intelligent_dq_agent.run(self.df)
        self.tool_outputs["DQ_Assessment"] = res
        return res

    def _governance_tool(self, _input: str = ""):
        """
        Governance consumes the DQ assessment output (artifact + llm_summary).
        Ensure IntelligentDQAgent returns 'artifact' (DataFrame) with a 'DQ_Score' column
        and 'llm_summary' (string) or similar summary in the returned dict.
        """
        dq_res = self._dq_assessment_tool("")

        artifact_df = dq_res.get("artifact")
        dataset_score = None
        if artifact_df is not None and hasattr(artifact_df, "columns") and "DQ_Score" in artifact_df.columns:
            try:
                dataset_score = float(artifact_df["DQ_Score"].mean())
            except Exception:
                dataset_score = None

        context = {
            "artifact": artifact_df,
            "llm_summary": dq_res.get("llm_summary"),
            "dataset_score": dataset_score,
            "correlation_matrix": dq_res.get("correlation_matrix", {})
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

    # ---------------- Router ----------------
    def route(self, user_query: str, callbacks=None):
        self.tool_outputs = {}  # reset for each run
        result = self.agent_executor.run(user_query, callbacks=callbacks)
        return {
            "llm_response": result if isinstance(result, str) else str(result),
            "tool_outputs": self.tool_outputs,
        }
