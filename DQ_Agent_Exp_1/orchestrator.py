# orchestrator.py
# orchestrator.py
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from llm_utils import get_llm
import os

from agents.profiling import profiling_agent
from agents.dq_scoring import dq_scoring_agent
from agents.rule_generation import rule_generation_agent
from agents.eda import eda_agent
from agents.text2df import text2df_agent
from agents.dq_insights import DQInsightsAgent

os.makedirs("outputs", exist_ok=True)

class Orchestrator:
    def __init__(self, df, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.df = df
        self.llm = get_llm(model=model, temperature=temperature)

        # instantiate insights agent with eda injected
        self.dq_insights = DQInsightsAgent(eda_agent)

        # Build tools: each tool is a wrapper returning strings (for LLM)
        tools = [
            Tool(
                name="profiling",
                func=self._profiling_tool,
                description="Run dataset profiling. Returns text summary and preview table."
            ),
            Tool(
                name="dq_scoring",
                func=self._dq_scoring_tool,
                description="Compute data quality scoring. Returns summary and preview table."
            ),
            Tool(
                name="rule_generation",
                func=self._rule_generation_tool,
                description="Generate business rules from profiling + dq scoring."
            ),
            Tool(
                name="dq_insights",
                func=self._dq_insights_tool,
                description="Produce remediation suggestions and trigger EDA on worst columns."
            ),
            Tool(
                name="text2df",
                func=self._text2df_tool,
                description="Convert NL to SQL and query the dataframe."
            )
        ]

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True
        )

        self.agent_executor = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory
        )

    # ---------------- RAW methods (return dicts for internal use) ----------------
    def _profiling_raw(self):
        return profiling_agent.run(self.df)

    def _dq_scoring_raw(self):
        return dq_scoring_agent.run(self.df)

    # ---------------- TOOL methods (return strings for LLM) ----------------
    def _profiling_tool(self, _input: str = ""):
        res = self._profiling_raw()
        preview_df = res["summary_df"].head(5).to_markdown(index=False)
        return (
            f"[ProfilingAgent] Profiling results saved to {res['summary_path']}.\n\n"
            f"Quick Summary: {res['text_summary']}\n\n"
            f"Table Preview:\n{preview_df}"
        )

    def _dq_scoring_tool(self, _input: str = ""):
        res = self._dq_scoring_raw()
        preview_df = res["dq_df"].head(5).to_markdown(index=False)
        return (
            f"[DQScoringAgent] Scoring results saved to {res['dq_path']}.\n\n"
            f"Preview:\n{preview_df}"
        )

    def _rule_generation_tool(self, _input: str = ""):
        profiling = self._profiling_raw()
        dq = self._dq_scoring_raw()
        context = profiling["text_summary"] + "\n" + dq["summary_text"]
        return rule_generation_agent.run(context=context, df=self.df)

    def _dq_insights_tool(self, _input: str = ""):
        profiling = self._profiling_raw()
        dq = self._dq_scoring_raw()
        ctx = profiling["text_summary"] + "\n" + dq["summary_text"]
        return self.dq_insights.run(df=self.df, context=ctx, top_n=3)

    def _text2df_tool(self, user_query: str = ""):
        return text2df_agent.run(self.df, query=user_query)

    # ---------------- Public method ----------------
    def route(self, user_query: str):
        """Primary entry: forward to LangChain agent executor which will call tools as needed."""
        return self.agent_executor.run(user_query)
