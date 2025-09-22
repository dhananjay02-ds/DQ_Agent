# agents/eda.py
import os
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.makedirs("outputs", exist_ok=True)

class EDAAgent:
    def __init__(self):
        pass

    def run(self, df: pd.DataFrame, query: str = None, target_col: str = None) -> str:
        """
        Return base64 PNG string (truncated) and save full PNG to outputs/
        """
        fig = None
        if target_col and target_col in df.columns:
            plt.figure(figsize=(6,4))
            if pd.api.types.is_numeric_dtype(df[target_col]):
                sns.histplot(df[target_col].dropna(), kde=True)
            else:
                sns.countplot(y=df[target_col])
            plt.title(f"{target_col} distribution")
            fig = plt.gcf()
            out_file = os.path.join("outputs", f"eda_{target_col}.png")
        else:
            plt.figure(figsize=(8,6))
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) >= 2:
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
            else:
                plt.text(0.5, 0.5, "No numeric columns for correlation", ha='center')
            fig = plt.gcf()
            out_file = os.path.join("outputs", "eda_correlation.png")

        plt.tight_layout()
        fig.savefig(out_file)
        plt.close(fig)

        # return truncated base64 to keep response size reasonable
        with open(out_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return f"[EDAAgent] saved:{out_file} base64_trunc={encoded[:200]}..."

eda_agent = EDAAgent()
