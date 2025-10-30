scores = list(map(int, input().split()))
participants = []
for i in range(1, 32):
    name = ""
    score = 0
    if i & 1:
        name += "A"
        score += scores[0]
    if i & 2:
        name += "B"
        score += scores[1]
    if i & 4:
        name += "C"
        score += scores[2]
    if i & 8:
        name += "D"
        score += scores[3]
    if i & 16:
        name += "E"
        score += scores[4]
    participants.append((name, score))

participants.sort(key=lambda x: (-x[1], x[0]))

for name, score in participants:
    print(name)