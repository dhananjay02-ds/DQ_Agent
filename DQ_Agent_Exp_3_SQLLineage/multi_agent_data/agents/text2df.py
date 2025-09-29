# agents/text2df.py
import pandas as pd
from pandasql import sqldf
import re
from llm_utils import get_llm

class Text2DFAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def run(self, df: pd.DataFrame, query: str):
        schema = f"DataFrame columns: {df.columns.tolist()}"
        prompt = f"""
        Convert this natural language request into a SQL query for pandasql on DataFrame df.
        {schema}
        Request: {query}
        Only return SQL.
        """
        sql_query = self.llm.predict(prompt).strip()
        sql_query = re.sub(r"```(?:sql)?\n|```", "", sql_query).strip()

        try:
            result = sqldf(sql_query, {"df": df})
            return {
                "llm_summary": f"[Text2DFAgent] Ran query: {sql_query}",
                "artifact": result
            }
        except Exception as e:
            return {"llm_summary": f"[Text2DFAgent] Error: {e}", "artifact": None}

text2df_agent = Text2DFAgent()
