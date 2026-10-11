p = "scripts/build_nb.py"
lines = open(p, encoding="utf-8").read().split("\n")
out = []
skip = 0
for i, l in enumerate(lines):
    # skip the broken duplicate kpis = ( ... ) block
    if i == 40:  # 0-indexed: line 41
        skip = 13  # lines 41..53 inclusive (13 lines)
    elif skip > 0:
        skip -= 1
        continue
    # remove stray md( calls: line 245 (0-indexed 244) and line 246 (0-indexed 245)
    elif i == 244 or i == 245:
        continue
    # remove stray extra parentheses on line 252 (0-indexed 251)
    elif i == 251:
        if l.strip() == ")":
            continue
    out.append(l)
open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("patched; new line count:", len(out))
PYEOF