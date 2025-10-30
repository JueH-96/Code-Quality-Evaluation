n, m = map(int, input().split())
s = input().strip()

segments = []
current = []
for c in s:
    if c == '0':
        if current:
            segments.append(current)
            current = []
    else:
        current.append(c)
if current:
    segments.append(current)

max_logos = 0
for seg in segments:
    a = seg.count('1')
    b = seg.count('2')
    required = b + max(a - m, 0)
    if required > max_logos:
        max_logos = required

print(max_logos)