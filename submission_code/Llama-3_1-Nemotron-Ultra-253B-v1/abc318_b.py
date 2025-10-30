n = int(input())
rects = []
x_events = set()

for _ in range(n):
    a, b, c, d = map(int, input().split())
    rects.append((a, b, c, d))
    x_events.add(a)
    x_events.add(b)

x_list = sorted(x_events)
total = 0

for i in range(len(x_list) - 1):
    x1 = x_list[i]
    x2 = x_list[i + 1]
    width = x2 - x1
    if width == 0:
        continue
    
    active = []
    for (a, b, c, d) in rects:
        if a <= x1 and b >= x2:
            active.append((c, d))
    
    if not active:
        continue
    
    active.sort()
    merged = [active[0]]
    for c, d in active[1:]:
        last_c, last_d = merged[-1]
        if c <= last_d:
            merged[-1] = (last_c, max(last_d, d))
        else:
            merged.append((c, d))
    
    sum_length = sum(d - c for c, d in merged)
    total += width * sum_length

print(total)