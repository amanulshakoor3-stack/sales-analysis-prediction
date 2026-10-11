"""Rebuild the narrative notebook, execute it headlessly, and save it.

Usage:
    python scripts/execute_nb.py
Runs from the project root so `notebooks/` resolves correctly.
"""
import subprocess
import sys

import nbformat
from nbclient import NotebookClient

PROJECT = "C:/New D/project/sales-analysis-prediction"

# 1) Build the notebook source.
subprocess.run(
    [sys.executable, "scripts/build_nb.py"],
    check=True,
    cwd=PROJECT,
)

# 2) Execute it headlessly (nbclient, which works where nbconvert's kernel-spawn
#    crashed) and save in place.
nb = nbformat.read(f"{PROJECT}/notebooks/sales_analysis.ipynb", as_version=4)
client = NotebookClient(
    nb,
    timeout=300,
    kernel_name="python3",
    resources={"metadata": {"path": f"{PROJECT}/notebooks/"}},
)
client.execute()
nbformat.write(nb, f"{PROJECT}/notebooks/sales_analysis.ipynb")
print("EXECUTED OK -> notebooks/sales_analysis.ipynb")
