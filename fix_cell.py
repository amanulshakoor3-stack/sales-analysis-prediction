"""Rewrite the corrupted imports cell in scripts/build_nb.py."""
import sys

p = "C:/New D/project/sales-analysis-prediction/scripts/build_nb.py"
with open(p, encoding="utf-8") as fh:
    lines = fh.read().split("\n")

# Locate the mangled code(...) call block for the imports cell.
start = None
for i, l in enumerate(lines):
    if l.strip() == "code(" and i + 1 < len(lines) and lines[i + 1].strip().startswith('"""import sys'):
        start = i
        break
if start is None:
    print("ERROR: could not find the mangled code cell")
    sys.exit(1)

end = None
for i, l in enumerate(lines):
    if 'print("Columns:", list(clean.columns))"""' in l:
        end = i
        break
if end is None:
    print("ERROR: could not find the cell end")
    sys.exit(1)

clean_block = [
    "code(",
    '    """import sys',
    "import pathlib",
    'sys.path.insert(0, str(pathlib.Path("..").resolve()))',
    "",
    "import numpy as np",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "%matplotlib inline",
    "from data import generate_sales_data, clean_data",
    "",
    'sns.set_theme(style="whitegrid")',
    'plt.rcParams["figure.dpi"] = 100',
    "",
    "raw = generate_sales_data(n=6000, seed=42)",
    "clean = clean_data(raw)",
    'print("Generated", len(raw), "line items")',
    'print("After cleaning:", len(clean), "rows")',
    'print("Nulls / duplicates:", int(clean.isna().sum().sum()), "/",',
    "      int(clean.duplicated().sum()))",
    'print("Columns:", list(clean.columns)))"""',
    ")",
]

lines[start : end + 2] = clean_block
out = "\n".join(lines)
with open(p, "w", encoding="utf-8") as fh:
    fh.write(out)
print("rewrote imports cell; file now has", len(lines), "lines")
