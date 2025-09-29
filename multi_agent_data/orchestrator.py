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

os.makedirs("outputs", exist_ok=True)


class Orchestrator:
    def __init__(self, df, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.df = df
        self.llm = get_llm(model=model, temperature=temperature)

        # Track outputs for Streamlit
        self.tool_outputs = {}

        # Agents
        self.profiling_agent = ProfilingAgent()
        self.advanced_profiling_agent = AdvancedProfilingAgent()
        self.insights_agent = InsightsAgent()
        self.governance_agent = GovernanceAgent(eda_agent)
        self.text2df_agent = text2df_agent
        self.cross_relation_agent = CrossRelationAgent()

        # Tools
        tools = [
            Tool(name="profiling", func=self._profiling_tool,
                 description="Run dataset profiling (dashboard-based)."),
            Tool(name="advanced_profiling", func=self._advanced_profiling_tool,
                 description="Perform advanced profiling with anomalies, drift, correlations, dataset score."),
            Tool(name="insights", func=self._insights_tool,
                 description="Analyze advanced profiling results column-wise and dataset-level."),
            Tool(name="governance", func=self._governance_tool,
                 description="Consume profiling + insights to suggest governance rules, issues, remediation."),
            Tool(name="text2df", func=self._text2df_tool,
                 description="Convert NL to SQL and query the dataframe."),
            Tool(
                name="cross_relation",
                func=self._cross_relation_tool,
                description="Check consistency of relationships between categorical column pairs."
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

    # ---------------- Router ----------------
    def route(self, user_query: str, callbacks=None):
        self.tool_outputs = {}  # reset for each run
        result = self.agent_executor.run(user_query, callbacks=callbacks)
        return {
            "llm_response": result if isinstance(result, str) else str(result),
            "tool_outputs": self.tool_outputs,
        }
