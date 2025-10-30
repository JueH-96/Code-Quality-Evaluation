a, b, c, d, e = map(int, input().split())

participants = []
for mask in range(1, 32):
    name = []
    score = 0
    for i in range(5):
        if (mask >> i) & 1:
            name.append('ABCDE'[i])
            score += [a, b, c, d, e][i]
    participants.append((score, ''.join(name)))

participants.sort(key=lambda x: (-x[0], x[1]))

for p in participants:
    print(p[1])