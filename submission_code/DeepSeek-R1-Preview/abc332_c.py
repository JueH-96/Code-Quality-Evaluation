n, m = map(int, input().split())
s = input().strip()

periods = []
current_period = []

for ch in s:
    if ch == '0':
        if current_period:
            periods.append(current_period)
            current_period = []
    else:
        current_period.append(ch)
if current_period:
    periods.append(current_period)

max_logo = 0

for p in periods:
    c1 = p.count('1')
    c2 = p.count('2')
    needed_logo = c2 + max(c1 - m, 0)
    if needed_logo > max_logo:
        max_logo = needed_logo

print(max_logo)