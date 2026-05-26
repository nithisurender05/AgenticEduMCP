# AgenticEduMCP Research

## Overview

This repository contains research code for studying agentic large language models (LLMs) in personalized education. The system uses an MCP-based context retrieval server to provide student-specific Canvas-style data to LLMs and evaluates how tool-enabled, contextual, and baseline responses compare.

## Key Components

- `canvas_mcp_server.py` - Mock MCP server exposing student grade history, instructor feedback, and course syllabus via tool APIs.
- `run_single_experiment.py` - Executes an experiment for one student using Gemini LLMs, comparing baseline, contextual, and agentic tool-enabled tutoring.
- `run_all.py` - Runs the experiment across all mock students and stores aggregated results in `results.json`.
- `scoring.py` - Evaluates generated responses using heuristic scoring and a GPT-based judge prompt.
- `analyze_and_plot.py` - Processes `evaluation_results.json` and generates visualization figures (`fig1_overall.png`, `fig2_metrics.png`).
- `mock_canvas_data.json` - Synthetic Canvas-style student profiles, course grades, feedback, and syllabus data.
- `evaluation_results.json` - Precomputed evaluation records used by analysis and plotting.

## Setup

1. Create and activate a Python virtual environment.
2. Install required packages (assuming dependencies are available in your environment).
3. Set the Gemini API key:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

## Usage

- Start the mock MCP server:

```bash
python3 canvas_mcp_server.py
```

- Run a single experiment for all students:

```bash
python3 run_all.py
```

- Score the experiment outputs and generate evaluation metrics:

```bash
python3 scoring.py
```

- Create analysis figures:

```bash
python3 analyze_and_plot.py
```

## Research Focus

This project explores:

- Agentic LLM interactions using MCP tool invocation
- Contextual personalization in educational guidance
- Comparative evaluation of baseline, contextual, and agentic tutoring methods
- Metrics such as grounding, personalization, specificity, and actionability

## Notes

- The script `run_single_experiment.py` uses Google Gemini models and requires `GEMINI_API_KEY`.
- The MCP server is implemented with a mock Canvas dataset and can be extended for richer student or course context.
- Figures created by `analyze_and_plot.py` are intended for research reporting.
