import os
import re
import pandas as pd
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from llm_utils import get_llm

# --- Agents ---
from agents.profiling import ProfilingAgent
from agents.intelligent_profiler import IntelligentDQAgent
from agents.governance import GovernanceAgent
from agents.text2df import text2df_agent
from agents.eda import eda_agent
from agents.semantic_validator_agent import SemanticValidatorAgent
from agents.causal import CausalAgent  
from agents.visualization_agent import LLMVisualizationAgent
from agents.drift_agent import DriftAgent

os.makedirs("outputs", exist_ok=True)


class Orchestrator:
    """
    Central orchestrator coordinating profiling, data quality, governance,
    semantic validation, and causal inference agents.
    """

    def __init__(self, df, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.df = df
        self.llm = get_llm(model=model, temperature=temperature)
        self.tool_outputs = {}

        # ------------------ Agents ------------------
        self.profiling_agent = ProfilingAgent()
        self.intelligent_dq_agent = IntelligentDQAgent()
        self.governance_agent = GovernanceAgent(eda_agent)
        self.text2df_agent = text2df_agent
        self.semantic_validator_agent = SemanticValidatorAgent()
        self.causal_agent = CausalAgent()  
        self.visualization_agent = LLMVisualizationAgent()
        self.drift_agent = DriftAgent()

        # ------------------ Tools ------------------
        tools = [
            Tool(
                name="profiling",
                func=self._profiling_tool,
                description='''Generate visual dashboards and statistical summaries of the dataset using YData-profiling or similar libraries.
This tool is only for visualization or demo purposes and should not be used for reasoning or validation tasks.
Example: Show a profiling dashboard of the uploaded dataset.'''
            ),
            Tool(
                name="dq_assessment",
                func=self._dq_assessment_tool,
                description='''Perform a quantitative data quality analysis over the entire dataset.
It computes completeness, consistency, uniqueness, and outlier drifts, and generates a Data Quality Score.
This tool is meant for evaluating dataset reliability, not semantic or logical checks.
Example: How good is my dataset? or Give me the data quality score.'''
            ),
            Tool(
                name="semantic_validation",
                func=self._semantic_validator_tool,
                description='''Detect semantic or logical inconsistencies between columns in the dataset.
This includes checking cross-column contradictions, mismatched logic, or data misclassifications (e.g., fund type not matching fund name).
It ensures meaning-level correctness, not statistical quality.
Example: Are there logical inconsistencies in FundType?'''
            ),
            # Tool(
            #     name="causal_analysis",
            #     func=self._causal_tool,
            #     description="Run causal inference to identify causal relationships."
            # ),
            Tool(
                name="governance",
                func=self._governance_tool,
                description='''Translate data quality or semantic findings into governance rules and business policies.
It provides recommendations on stewardship, ownership, and process controls.
This tool should be used only when explicitly asked to generate or refine governance policies — not automatically after semantic or DQ checks.
Example: Suggest governance rules based on detected DQ issues.'''
            ),
            Tool(
                name="text2df",
                func=self._text2df_tool,
                description='''Convert natural language queries into SQL-like operations on the active DataFrame.
It allows users to fetch or filter data, compute aggregates, or summarize subsets (e.g., total assets by channel from June to August in Japan).
Use this tool whenever the query involves extracting, aggregating, comparing data or performing any action that can be solved using SQL query.
Example: Show me top 5 funds by growth rate.'''
            ),
            Tool(
            name="visualization",
            func=self._visualization_tool,
            description='''Generate intelligent, context-aware visual plots from natural language queries.
                Supports trend, comparison, correlation, and single-column distribution charts.'''
            ),
            Tool(
            name="drift_analysis",
            func=self._drift_tool,
            description='''Detect and visualize drift in dataset columns over time or between slicesusing PSI, KS-test, and chi-square metrics.'''
            ),
        ]

        # ------------------ Memory + Agent ------------------
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            output_key="output",
            return_messages=True
        )

        self.agent_executor = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory
        )

    # ------------------ Tool Wrappers ------------------
    def _profiling_tool(self, _input: str = ""):
        res = self.profiling_agent.run(self.df)
        self.tool_outputs["Profiling"] = res
        return res

    def _dq_assessment_tool(self, _input: str = ""):
        res = self.intelligent_dq_agent.run(self.df)
        self.tool_outputs["DQ_Assessment"] = res
        return res

    def _semantic_validator_tool(self, user_query: str): 
        res = self.semantic_validator_agent.run(self.df, query=user_query)
        self.tool_outputs["SemanticValidation"] = res
        return res

    def _causal_tool(self, user_query: str = ""):
        res = self.causal_agent.run(self.df, query=user_query)
        self.tool_outputs["CausalAnalysis"] = res
        return res

    def _governance_tool(self, _input: str = ""):
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

    def _visualization_tool(self, user_query: str):
        res = self.visualization_agent.run(self.df, query=user_query)
        self.tool_outputs["VISUALIZATION"] = res
        return res

    def _drift_tool(self, user_query: str):
        if getattr(self, "df", None) is None or self.df.empty:
            return {"summary": "No active dataset found for drift analysis.", "artifact": None, "table": pd.DataFrame()}
        try:
            result = self.drift_agent.run(self.df, query=user_query)
            self.tool_outputs["DRIFT"] = result
        except Exception as e:
            result = {"summary": f"Drift analysis failed: {e}", "artifact": None, "table": pd.DataFrame()}
        return result


    def route(self, user_query: str, callbacks=None):
        """
        Routes the user query through all specialized agents and synthesizes:
        1. 💭 Thought — a structured reasoning summary (step-by-step).
        2. ✅ Final Answer — a concise, factual, business-ready statement.
        """
    
        self.tool_outputs = {}
    
        # --- Step 1: Run reasoning chain ---
        base_result = self.agent_executor.run(user_query, callbacks=callbacks)
        base_response = base_result if isinstance(base_result, str) else str(base_result)
    
        # --- Step 2: Aggregate tool-level summaries ---
        tool_steps = []
        for name, out in self.tool_outputs.items():
            if isinstance(out, dict) and out.get("llm_summary"):
                tool_steps.append(f"- **{name}** → {out['llm_summary']}")
            else:
                tool_steps.append(f"- **{name}** → completed successfully.")
        tool_steps_text = "\n".join(tool_steps)
    
        # --- Step 3: Capture last tool for factual grounding ---
        last_tool_name, last_output, last_summary = None, None, None
        if self.tool_outputs:
            last_tool_name = list(self.tool_outputs.keys())[-1]
            last_output = self.tool_outputs[last_tool_name]
            if isinstance(last_output, dict):
                last_summary = last_output.get("llm_summary", "")
                artifact = last_output.get("artifact", None)
            else:
                artifact = last_output
        else:
            artifact, last_summary = None, None
    
        # --- Step 4: Prepare factual data for grounding ---
        artifact_preview = ""
        if isinstance(artifact, pd.DataFrame):
            try:
                if artifact.shape[0] <= 10 and artifact.shape[1] <= 5:
                    artifact_preview = "\n\nLast tool produced this table:\n" + artifact.to_string(index=False)
                else:
                    artifact_preview = f"\n\nLast tool returned a DataFrame with shape {artifact.shape}."
            except Exception:
                pass
        elif isinstance(artifact, (float, int)):
            artifact_preview = f"\n\nNumeric result from last tool: {artifact:.2f}"
        elif isinstance(artifact, str) and len(artifact) < 400:
            artifact_preview = f"\n\nText output from last tool:\n{artifact}"
    
        # --- Step 5: Structured synthesis prompt ---
        synthesis_prompt = f"""
        You are a senior data reasoning AI working inside a multi-agent orchestration system.
        The user asked: "{user_query}"
        
        Here are the ordered reasoning steps and their summaries:
        {tool_steps_text}
        
        The last executed tool was: {last_tool_name or "Unknown"}
        Summary: {last_summary or "No summary."}
        {artifact_preview}
        
        Now, synthesize the final reasoning outcome.
        
        Write **two distinct sections**:
        1. **Thought:** Provide a clear, step-by-step structured reasoning log of how the agents approached this.
           - Write it as short bullet points or numbered steps.
           - Be factual and modular, not narrative prose.
           - Reference what each agent did and how the reasoning evolved.
        
        2. **Final Answer:** Give a crisp, factual, and business-facing result — no essays.
           - State the numeric or concrete output clearly if available.
           - Avoid redundancy or rephrasing of process.
           - If relevant, end with: “📎 Artifacts from this step are shown below.”
        """
    
        # --- Step 6: Generate synthesis via LLM ---
        synthesis = self.llm.invoke(synthesis_prompt)
        synthesis_text = getattr(synthesis, "content", str(synthesis))
    
        # --- Step 7: Extract Thought & Final Answer ---
        thought, final_answer = None, None
        match = re.search(r"Thought:(.*)Final Answer:(.*)", synthesis_text, re.S | re.I)
        if match:
            thought = match.group(1).strip()
            final_answer = match.group(2).strip()
        else:
            parts = re.split(r"\n{2,}", synthesis_text.strip(), maxsplit=1)
            if len(parts) == 2:
                thought, final_answer = parts[0].strip(), parts[1].strip()
            else:
                final_answer = synthesis_text.strip()
    
        # --- Step 8: Append artifact notice if not mentioned ---
        if isinstance(final_answer, str) and "Artifacts" not in final_answer and artifact is not None:
            final_answer += "\n\n📎 Artifacts from this step are shown below."
    
        return {
            "tool_outputs": self.tool_outputs,
            "final_thought": thought,
            "llm_response": final_answer or base_response,
        }
