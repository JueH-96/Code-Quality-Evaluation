n, m = map(int, input().split())
s = input().strip()

segments = []
current = []

for c in s:
    if c == '0':
        if current:
            segments.append(''.join(current))
            current = []
    else:
        current.append(c)
if current:
    segments.append(''.join(current))

max_logo = 0

for seg in segments:
    cnt_2 = seg.count('2')
    cnt_1 = seg.count('1')
    required = cnt_2 + max(0, cnt_1 - m)
    if required > max_logo:
        max_logo = required

print(max_logo)