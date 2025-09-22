# agents/text2df.py
import os
import re
import pandas as pd
from pandasql import sqldf
from llm_utils import get_llm

os.makedirs("outputs", exist_ok=True)

class Text2DFAgent:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = get_llm(model=model)

    def _strip_code_fence(self, t: str) -> str:
        return re.sub(r"```(?:sql)?\n?|```", "", t).strip()

    def run(self, df: pd.DataFrame, query: str = None, target_col: str = None):
        if query is None:
            return {"error": "No query provided"}
        columns = df.columns.tolist()
        prompt = f"Convert the following natural language request into a SQL query for pandas DataFrame named df.\nColumns: {columns}\nRequest: {query}\nReturn only SQL."
        resp = self.llm.predict(prompt)
        sql = self._strip_code_fence(resp)
        try:
            result = sqldf(sql, {"df": df})
            if isinstance(result, pd.DataFrame):
                out_path = os.path.join("outputs", "text2df_result.csv")
                result.to_csv(out_path, index=False)
                return {"result_path": out_path, "result_df": result}
            else:
                return {"scalar": result}
        except Exception as e:
            return {"error": str(e), "sql_used": sql}
        
text2df_agent = Text2DFAgent()
