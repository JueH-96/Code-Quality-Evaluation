a, b, c, d, e = map(int, input().split())

participants = []

for i in range(1, 32):
    name = []
    total = 0
    for j in range(5):
        if (i >> j) & 1:
            name.append('ABCDE'[j])
            total += [a, b, c, d, e][j]
    name_str = ''.join(name)
    participants.append((total, name_str))

participants.sort(key=lambda x: (-x[0], x[1]))

for p in participants:
    print(p[1])