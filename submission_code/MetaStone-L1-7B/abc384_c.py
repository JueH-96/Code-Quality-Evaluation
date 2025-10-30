a, b, c, d, e = map(int, input().split())

participants = []

for mask in range(1, 32):
    s = 0
    name = ''
    for i in range(5):
        if mask & (1 << i):
            s += [a, b, c, d, e][i]
            name += chr(ord('A') + i)
    participants.append((s, name))

participants.sort(key=lambda x: (-x[0], x[1]))

for p in participants:
    print(p[1])