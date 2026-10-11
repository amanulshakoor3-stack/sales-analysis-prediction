# Build the corrected kpis definition and replace the broken block in build_nb.py.
p = "scripts/build_nb.py"
src = open(p, encoding="utf-8").read()
lines = src.split(chr(10))

start = None
for i, l in enumerate(lines):
    if l.strip() == "kpis = (":
        start = i
        break
if start is None:
    print("kpis not found")
    raise SystemExit(1)

end = start
for j in range(start, len(lines)):
    if lines[j].strip() == ")":
        end = j
        break

BS = chr(92)      # backslash
DQ = chr(34)      # double quote
NL = chr(10)      # newline

# Build the new kpis definition lines.
new_lines = [
    "kpis = " + DQ + BS + "n" + DQ + ".join([",
    '    f"- **Total revenue:** ${clean[\'line_total\'].sum():,.0f}",',
    '    f"- **Total gross profit:** ${clean[\'gross_profit\'].sum():,.0f}",',
    '    f"- **Average order value:** ${clean[\'line_total\'].mean():,.0f}",',
    '    f"- **Unique regions:** {clean[\'region\'].nunique()} | **Unique categories:** {clean[\'category\'].nunique()}",',
    '    f"- **Top region:** {clean.groupby(\'region\')['line_total'].sum().idxmax()} (${clean.groupby(\'region\')['line_total'].sum().max():,.0f})",',
    '    f"- **Top category:** {clean.groupby(\'category\')['line_total'].sum().idxmax()} (${clean.groupby(\'category\')['line_total'].sum().max():,.0f})",',
    "])",
]

lines[start:end+1] = new_lines
open(p, "w", encoding="utf-8").write(chr(10).join(lines))
print("patched kpis at line", start+1)
