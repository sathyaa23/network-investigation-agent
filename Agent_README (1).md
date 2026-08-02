# Network Data Investigation Agent

An AI agent that autonomously investigates network-session data for anomalies and
fraud, such as data exfiltration and brute-force logins. Given an investigation goal,
it plans an approach, calls analysis tools, observes the results, and reasons across
multiple steps until it can produce an evidence-based findings report.

Built with Python and the Anthropic API's tool-calling (function-calling) interface.

## Overview

A single language-model call answers a prompt once. An agent runs a loop: the model
decides which tool to invoke, the program executes it, the result is fed back, and the
model decides the next step, continuing until it has enough evidence to answer. This
project implements that loop directly rather than relying on a higher-level framework,
making the underlying mechanism explicit.

```
goal -> model -> needs a tool? - yes -> run tool -> return result -+
                    |                                              |
                    +- no -> final report                         |
                    ^                                              |
                    +----------------------------------------------+
```

## Features

- Autonomous multi-step investigation driven by an LLM
- Four analysis tools the agent selects from on its own:
  - `list_columns` — inspect the dataset schema
  - `query_sessions` — filter sessions by user, bytes, port, failed logins, country
  - `compute_stats` — summary statistics for a column
  - `detect_anomalies` — flag outlier rows via z-score or IQR
- A synthetic network-session dataset with planted anomalies (a data-exfiltration user
  and a brute-force user) for demonstration
- A step guard to bound the agent loop
- Step-by-step output showing the agent's reasoning and tool calls

## Installation

```bash
pip install anthropic pandas numpy
export ANTHROPIC_API_KEY=your_key_here
```

## Usage

```bash
python network_agent.py "Investigate whether any users show signs of data exfiltration or brute-force logins, and report the top suspicious sessions."
```

The agent prints each step as it reasons and calls tools, followed by a final report
identifying the suspicious users, the supporting evidence (session IDs and metrics),
and a recommended action.

Set the `MODEL` constant near the top of `network_agent.py` to a model available on
your account. The same pattern applies to OpenAI's function-calling API.

## How it works

- **Agent loop** (`run_agent`): sends the conversation and tool schemas to the model.
  When the model requests a tool (`stop_reason == "tool_use"`), the program executes
  the tool, appends the result as a `tool_result` message, and loops. When the model
  responds without a tool call, its text is returned as the final report.
- **Tools**: plain Python functions over a pandas DataFrame. The model never executes
  code itself; it requests a tool by name with arguments, and the program runs it.
- **Tool schemas**: JSON schemas describe each tool's name, purpose, and inputs so the
  model knows how to call them.
- **System prompt**: defines the agent's role, what to look for, and when to stop and
  write the report.

## Tech stack

Python, Anthropic API (tool-calling), pandas, NumPy.

## Possible extensions

- Add a planner/analyst multi-agent handoff
- Add trajectory logging and tracing of every step and tool call
- Orchestrate the loop with LangGraph and compare approaches
- Add an evaluation harness that verifies the agent recovers the planted anomalies

## License

MIT
