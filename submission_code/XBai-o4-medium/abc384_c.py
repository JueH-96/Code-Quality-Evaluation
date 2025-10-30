a, b, c, d, e = map(int, input().split())
letters = ['A', 'B', 'C', 'D', 'E']
values = [a, b, c, d, e]

participants = []

for mask in range(1, 32):
    name = ''
    score = 0
    for i in range(5):
        if mask & (1 << i):
            name += letters[i]
            score += values[i]
    participants.append((name, score))

sorted_participants = sorted(participants, key=lambda x: (-x[1], x[0]))

for p in sorted_participants:
    print(p[0])