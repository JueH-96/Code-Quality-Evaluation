n, m = map(int, input().split())
s = input().strip()

segments = []
current = []
for c in s:
    if c != '0':
        current.append(c)
    else:
        if current:
            segments.append(current)
            current = []
if current:
    segments.append(current)

max_required = 0
for seg in segments:
    c1 = seg.count('1')
    c2 = seg.count('2')
    required = c2 + max(0, c1 - m)
    if required > max_required:
        max_required = required

print(max_required if max_required > 0 else 0)