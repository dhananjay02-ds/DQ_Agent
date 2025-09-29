# main.py
import sys
import os
import pandas as pd
from orchestrator import Orchestrator

def run_flow(csv_path: str, query: str):
    df = pd.read_csv(csv_path)
    orch = Orchestrator(df)
    print("[INFO] Running orchestration...")
    result = orch.route(query)
    print("\n=== FINAL RESULT ===\n")
    print(result)
    print("\n[INFO] Check outputs/ for artifacts (profiling_summary.csv, dq_scoring.csv, eda images, etc).")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in environment before running.")
        sys.exit(1)

    if len(sys.argv) < 3:
        print("Usage: python main.py <path_to_csv> '<query>'")
        sys.exit(1)

    run_flow(sys.argv[1], sys.argv[2])
