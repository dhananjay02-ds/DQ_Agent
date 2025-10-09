import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io, base64

class EDAAgent:
    def run(self, df: pd.DataFrame, target_col: str = None):
        plt.figure(figsize=(6, 4))
        if target_col and target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                sns.histplot(df[target_col].dropna(), kde=True)
                plt.title(f"Distribution of {target_col}")
            else:
                sns.countplot(y=df[target_col])
                plt.title(f"Value counts of {target_col}")
        else:
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return encoded

eda_agent = EDAAgent()