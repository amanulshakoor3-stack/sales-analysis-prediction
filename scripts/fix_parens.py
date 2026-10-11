"""Balance the parentheses on the 'Columns' data-generation line."""
p = "scripts/build_nb.py"
with open(p, encoding="utf-8") as fh:
    lines = fh.read().split("\n")

fixed = False
for i, l in enumerate(lines):
    if '"Columns:"' in l and "clean.columns" in l:
        cell = l.split('"""')[0]
        opens = cell.count("(")
        closes = cell.count(")")
        removed = 0
        while closes > opens:
            idx = cell.rfind(")")
            cell = cell[:idx] + cell[idx + 1:]
            closes -= 1
            removed += 1
        lines[i] = cell + ('"""' if l.endswith('"""') else '')
        fixed = True
        print("fixed line", i + 1, "removed", removed, "paren(s) ->", repr(lines[i]))

if not fixed:
    print("NO LINE FOUND")

with open(p, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
