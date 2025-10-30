s = input().strip()
pos = [i for i, c in enumerate(s) if c == '|']
if len(pos) < 2:
    print("")
else:
    substring = s[pos[0] + 1:]
    parts = substring.split('|')
    parts = [p for p in parts if p]
    A = [len(p) for p in parts]
    print(' '.join(map(str, A)))