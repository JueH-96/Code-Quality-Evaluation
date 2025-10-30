n, m = map(int, input().split())
s = input().strip()

intervals = []
current = []
for c in s:
    if c == '0':
        if current:
            intervals.append(current)
            current = []
    else:
        current.append(c)
if current:
    intervals.append(current)

max_logo = 0
for interval in intervals:
    a = interval.count('2')
    b = interval.count('1')
    required = a + max(0, b - m)
    if required > max_logo:
        max_logo = required

print(max_logo)