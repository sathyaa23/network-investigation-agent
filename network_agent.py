"""
Network Data Investigation Agent (Agentic AI)
=============================================
An AI agent that autonomously investigates a dataset of network sessions for signs
of anomalous / fraudulent activity (data exfiltration, brute-force logins), by
PLANNING, CALLING TOOLS, OBSERVING results, and REASONING over multiple steps until
it can write a findings report.

This is a real agentic system, not a single LLM call. It implements the core agent
loop yourself, so you understand exactly what frameworks like LangChain / LangGraph
wrap:

        user goal
           │
           ▼
   ┌──► send messages + tool schemas to the LLM
   │        │
   │        ▼
   │   LLM decides: call a tool  ── yes ──► execute tool, return result ──┐
   │        │                                                             │
   │        └── no (final answer) ──► print report, stop                  │
   │                                                                      │
   └──────────────────────────────────────────────────────────────────◄──┘

Tools the agent can choose from:
  - list_columns()                      : see the dataset schema
  - query_sessions(filters)             : pull rows matching filters
  - compute_stats(column)               : summary statistics for a column
  - detect_anomalies(column, method)    : flag outlier rows (z-score or IQR)

The agent is given a goal and figures out WHICH tools to call, in what order, on its
own. That autonomy is what makes it "agentic."

Run it
------
    pip install anthropic pandas numpy
    export ANTHROPIC_API_KEY=sk-ant-...            # your key
    python network_agent.py "Investigate whether any users show signs of data \
        exfiltration or brute-force logins, and report the top suspicious sessions."

(OpenAI works the same way with their function-calling API; this uses Anthropic.)
"""

from __future__ import annotations
import json
import os
import sys
import random

import numpy as np
import pandas as pd
import anthropic

# Set this to a model you have access to (check current names in Anthropic docs).
MODEL = "claude-sonnet-4-5-20250929"
MAX_STEPS = 12


# ---------------------------------------------------------------------------
# 1. Synthetic dataset  (stands in for a real network-session / telecom table)
# ---------------------------------------------------------------------------
def build_dataset(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)
    rows = []
    for i in range(n):
        user = f"u{rng.randint(1, 40):03d}"
        bytes_sent = int(abs(np.random.normal(50_000, 20_000)))
        duration = round(abs(np.random.normal(120, 40)), 1)
        failed_logins = rng.choices([0, 1, 2], weights=[0.9, 0.08, 0.02])[0]
        rows.append({
            "session_id": f"s{i:04d}",
            "user_id": user,
            "bytes_sent": bytes_sent,
            "duration_s": duration,
            "dest_port": rng.choice([80, 443, 22, 3389, 8080]),
            "country": rng.choice(["US", "US", "US", "DE", "IN", "RU", "CN"]),
            "failed_logins": failed_logins,
        })
    df = pd.DataFrame(rows)

    # Plant a few real anomalies so the agent has something to find:
    # (a) data exfiltration: one user with huge byte transfers
    exfil_idx = df.sample(4, random_state=seed).index
    df.loc[exfil_idx, "bytes_sent"] = np.random.randint(800_000, 1_500_000, size=4)
    df.loc[exfil_idx, "user_id"] = "u007"
    # (b) brute force: one user with many failed logins
    bf_idx = df.sample(5, random_state=seed + 1).index
    df.loc[bf_idx, "failed_logins"] = np.random.randint(15, 40, size=5)
    df.loc[bf_idx, "user_id"] = "u013"
    df.loc[bf_idx, "dest_port"] = 22
    return df


DATA = build_dataset()


# ---------------------------------------------------------------------------
# 2. Tools  (plain Python functions the agent can call)
# ---------------------------------------------------------------------------
def list_columns() -> dict:
    return {"columns": list(DATA.columns), "row_count": len(DATA)}


def query_sessions(user_id=None, min_bytes=None, dest_port=None,
                   min_failed_logins=None, country=None, limit=10) -> dict:
    df = DATA
    if user_id is not None:
        df = df[df["user_id"] == user_id]
    if min_bytes is not None:
        df = df[df["bytes_sent"] >= min_bytes]
    if dest_port is not None:
        df = df[df["dest_port"] == dest_port]
    if min_failed_logins is not None:
        df = df[df["failed_logins"] >= min_failed_logins]
    if country is not None:
        df = df[df["country"] == country]
    return {"matched": len(df), "rows": df.head(int(limit)).to_dict(orient="records")}


def compute_stats(column: str) -> dict:
    if column not in DATA.columns:
        return {"error": f"unknown column '{column}'"}
    s = DATA[column]
    if not np.issubdtype(s.dtype, np.number):
        return {"value_counts": s.value_counts().to_dict()}
    return {
        "count": int(s.count()), "mean": round(float(s.mean()), 2),
        "std": round(float(s.std()), 2), "min": float(s.min()),
        "p95": round(float(s.quantile(0.95)), 2),
        "p99": round(float(s.quantile(0.99)), 2), "max": float(s.max()),
    }


def detect_anomalies(column: str, method: str = "zscore", threshold: float = 3.0) -> dict:
    if column not in DATA.columns:
        return {"error": f"unknown column '{column}'"}
    s = DATA[column]
    if not np.issubdtype(s.dtype, np.number):
        return {"error": f"column '{column}' is not numeric"}
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
    else:  # zscore
        z = (s - s.mean()) / (s.std() + 1e-9)
        mask = z.abs() > threshold
    flagged = DATA[mask]
    return {
        "method": method, "threshold": threshold, "num_flagged": int(mask.sum()),
        "flagged_rows": flagged.head(15).to_dict(orient="records"),
    }


TOOL_IMPL = {
    "list_columns": list_columns,
    "query_sessions": query_sessions,
    "compute_stats": compute_stats,
    "detect_anomalies": detect_anomalies,
}

# ---------------------------------------------------------------------------
# 3. Tool schemas  (what the LLM sees so it knows how to call each tool)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "list_columns",
        "description": "List the dataset's columns and row count. Call this first to orient.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_sessions",
        "description": "Return network sessions matching optional filters. Use to inspect specific users or suspicious patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "min_bytes": {"type": "integer"},
                "dest_port": {"type": "integer"},
                "min_failed_logins": {"type": "integer"},
                "country": {"type": "string"},
                "limit": {"type": "integer", "description": "max rows to return"},
            },
        },
    },
    {
        "name": "compute_stats",
        "description": "Summary statistics (mean, std, percentiles) for a numeric column, or value counts for a categorical column.",
        "input_schema": {
            "type": "object",
            "properties": {"column": {"type": "string"}},
            "required": ["column"],
        },
    },
    {
        "name": "detect_anomalies",
        "description": "Flag outlier rows in a numeric column using 'zscore' or 'iqr'. Use to find anomalous sessions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "method": {"type": "string", "enum": ["zscore", "iqr"]},
                "threshold": {"type": "number"},
            },
            "required": ["column"],
        },
    },
]

SYSTEM = (
    "You are a data-investigation agent for network security analytics. "
    "You are given an investigation goal and a set of tools over a table of network "
    "sessions. Plan your approach, call tools to gather evidence, and reason over the "
    "results across multiple steps. Look for data exfiltration (unusually large "
    "bytes_sent) and brute-force logins (high failed_logins). When you have enough "
    "evidence, STOP calling tools and write a concise findings report: the suspicious "
    "user(s), the specific evidence (numbers, session ids), and a recommended action. "
    "Be precise and state what the data does and does not show."
)


# ---------------------------------------------------------------------------
# 4. The agent loop  (this is the whole "agentic" mechanism)
# ---------------------------------------------------------------------------
def run_agent(goal: str) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    messages = [{"role": "user", "content": goal}]

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL, max_tokens=1500, system=SYSTEM, tools=TOOLS, messages=messages,
        )

        # Show the agent's reasoning/tool choices as it goes
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[step {step}] agent: {block.text.strip()}")
            elif block.type == "tool_use":
                print(f"[step {step}] -> calling tool: {block.name}({json.dumps(block.input)})")

        # If the model didn't ask for a tool, it's done — return its final text.
        if response.stop_reason != "tool_use":
            final = "".join(b.text for b in response.content if b.type == "text")
            return final

        # Otherwise execute every requested tool and feed results back.
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = TOOL_IMPL.get(block.name)
                try:
                    result = fn(**block.input) if fn else {"error": "unknown tool"}
                except Exception as e:
                    result = {"error": str(e)}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
        messages.append({"role": "user", "content": tool_results})

    return "Reached step limit before finishing the investigation."


def main():
    goal = " ".join(sys.argv[1:]) or (
        "Investigate whether any users show signs of data exfiltration or brute-force "
        "logins, and report the top suspicious sessions with evidence."
    )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY first. See README.")
        sys.exit(1)
    print(f"GOAL: {goal}")
    report = run_agent(goal)
    print("\n" + "=" * 60 + "\nFINAL REPORT\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
