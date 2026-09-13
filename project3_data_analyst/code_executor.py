"""
code_executor.py - Safely executes AI-generated analysis code
Runs the code in a controlled environment and captures output + charts.
"""
import sys
import io
import os
import json
import glob
import traceback
import pandas as pd
import numpy as np


def execute_analysis(code: str, df: pd.DataFrame, work_dir: str = ".") -> dict:
    """
    Execute AI-generated analysis code with the given dataframe.

    Returns:
        {
          "success": bool,
          "insights": list of strings,
          "charts": list of file paths,
          "stdout": str,
          "error": str or None
        }
    """
    # Clean up any existing chart files in work_dir
    for old_chart in glob.glob(os.path.join(work_dir, "chart_*.png")):
        try:
            os.remove(old_chart)
        except Exception:
            pass

    # Redirect stdout to capture print output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Change to work dir so charts save there
    old_cwd = os.getcwd()
    os.chdir(work_dir)

    result = {
        "success": False,
        "insights": [],
        "charts": [],
        "stdout": "",
        "error": None
    }

    try:
        # Build execution namespace
        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "df": df.copy(),
            "json": json,
        }

        # Execute the AI-generated code
        exec(code, exec_globals)

        # Capture output
        result["stdout"] = sys.stdout.getvalue()
        result["success"] = True

        # Extract insights from the INSIGHTS_JSON marker
        for line in result["stdout"].split("\n"):
            if line.startswith("INSIGHTS_JSON:"):
                try:
                    insights_raw = line[len("INSIGHTS_JSON:"):].strip()
                    result["insights"] = json.loads(insights_raw)
                except Exception:
                    pass

        # If no structured insights, extract any printed lines as fallback
        if not result["insights"]:
            lines = [l.strip() for l in result["stdout"].split("\n")
                     if l.strip() and not l.startswith("INSIGHTS_JSON")]
            result["insights"] = lines[:8]

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        result["stdout"] = sys.stdout.getvalue()

    finally:
        sys.stdout = old_stdout
        os.chdir(old_cwd)

    # Collect chart files
    result["charts"] = sorted(glob.glob(os.path.join(work_dir, "chart_*.png")))

    return result


def clean_code(raw_code: str) -> str:
    """Strip markdown fences if the AI included them."""
    code = raw_code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove first line (```python or ```) and last line (```)
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code
