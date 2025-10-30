scores = list(map(int, input().split()))
a, b, c, d, e = scores

participants = []
for i in range(32):
    name = ""
    score = 0
    if (i >> 0) & 1:
        name += "A"
        score += a
    if (i >> 1) & 1:
        name += "B"
        score += b
    if (i >> 2) & 1:
        name += "C"
        score += c
    if (i >> 3) & 1:
        name += "D"
        score += d
    if (i >> 4) & 1:
        name += "E"
        score += e
    if name:
        participants.append((score, name))

participants.sort(key=lambda x: (-x[0], x[1]))

for participant in participants:
    print(participant[1])