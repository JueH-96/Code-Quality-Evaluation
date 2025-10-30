a, b, c, d, e = map(int, input().split())
values = [a, b, c, d, e]
participants = []
for mask in range(1, 32):
    name = []
    total = 0
    for bit in range(4, -1, -1):
        if mask & (1 << bit):
            idx = 4 - bit
            name.append("ABCDE"[idx])
            total += values[idx]
    name_str = ''.join(name)
    participants.append((-total, name_str))
participants.sort()
for p in participants:
    print(p[1])