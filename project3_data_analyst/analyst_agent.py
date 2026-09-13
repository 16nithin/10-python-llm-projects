"""
analyst_agent.py - AI agent that analyzes CSV data and generates insights
Uses Groq API (free) to understand data and write analysis code.
"""
import requests
import json


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
SKIP_MODELS = ["whisper", "tts", "vision", "distil-whisper", "playai", "orpheus", "canopylabs", "guard", "rerank", "embed", "speech", "qwen"]


def get_best_model(api_key):
    """Fetch available Groq models and pick the best chat model."""
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if resp.status_code == 200:
            all_models = [m["id"] for m in resp.json().get("data", [])]
            chat_models = [m for m in all_models
                           if not any(skip in m.lower() for skip in SKIP_MODELS)]
            preferred = [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama3-70b-8192",
                "llama-3.1-8b-instant",
                "llama3-8b-8192",
                "gemma2-9b-it",
            ]
            for p in preferred:
                if p in chat_models:
                    return p
            if chat_models:
                return chat_models[0]
    except Exception as e:
        print(f"Could not fetch model list: {e}")
    return "gemma2-9b-it"


def call_groq(api_key, system_prompt, user_message, model, temperature=0.2, max_tokens=2000):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise ValueError(f"Groq API error {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"]


def get_data_summary(df):
    """Create a text summary of the dataframe for the AI."""
    lines = []
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append(f"\nColumns and types:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        lines.append(f"  - {col} ({dtype}), {nulls} nulls")

    lines.append(f"\nFirst 3 rows (as JSON):")
    lines.append(df.head(3).to_json(orient="records", indent=2))

    lines.append(f"\nNumeric column statistics:")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        lines.append(df[numeric_cols].describe().to_string())
    else:
        lines.append("  No numeric columns found.")

    return "\n".join(lines)


def generate_analysis_code(api_key, model, data_summary, user_question=""):
    """Ask the AI to write Python analysis code for the data."""

    system_prompt = """You are an expert Python data analyst.
Given a description of a CSV dataset, write Python code to analyze it and generate insights.

IMPORTANT RULES:
1. The dataframe is already loaded as `df` — do NOT load any files.
2. Use only: pandas, numpy, matplotlib, matplotlib.pyplot as plt
3. For each chart: always call plt.figure() first, then plt.tight_layout(), then plt.savefig(f'chart_N.png', dpi=100, bbox_inches='tight') and plt.close()
4. Use chart filenames: chart_1.png, chart_2.png, etc.
5. Store all text insights in a list called `insights` (list of strings)
6. At the end, print: INSIGHTS_JSON: followed by the JSON of the insights list
7. Keep charts simple and clear — bar, line, histogram, scatter, pie
8. Handle missing values gracefully (dropna() where needed)
9. Output ONLY the Python code — no explanation, no markdown, no ```python blocks

Example structure:
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

insights = []

# Analysis 1
...
insights.append("Finding: ...")

# Chart 1
plt.figure(figsize=(8, 5))
...
plt.tight_layout()
plt.savefig('chart_1.png', dpi=100, bbox_inches='tight')
plt.close()

# More analysis and charts...

import json
print("INSIGHTS_JSON:" + json.dumps(insights))"""

    question_part = f"\n\nUser's specific question: {user_question}" if user_question.strip() else ""

    user_message = f"""Here is the dataset description:

{data_summary}
{question_part}

Write Python analysis code following the rules above.
Generate 2-4 meaningful charts and 4-6 key insights about this data."""

    return call_groq(api_key, system_prompt, user_message, model, temperature=0.1, max_tokens=1800)


def generate_summary_report(api_key, model, data_summary, insights, user_question=""):
    """Ask the AI to write a plain-English report based on the insights."""

    system_prompt = """You are a friendly data analyst writing a clear report for business users.
Given data facts and insights, write a concise, readable report.

Format:
## Data Overview
(2-3 sentences about what the dataset contains)

## Key Findings
(4-6 bullet points, each starting with an emoji like 📈 📉 ⚠️ ✅)

## Recommendations
(2-3 actionable suggestions based on the data)

## Bottom Line
(1-2 sentences: the single most important takeaway)

Write in plain English. No jargon. Be specific with numbers when available."""

    question_part = f"\n\nUser's question was: {user_question}" if user_question.strip() else ""

    user_message = f"""Dataset description:
{data_summary}

Computed insights from the data:
{chr(10).join(f'- {i}' for i in insights)}
{question_part}

Write the report now."""

    return call_groq(api_key, system_prompt, user_message, model, temperature=0.3, max_tokens=1000)
