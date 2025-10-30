a, b, c, d, e = map(int, input().split())
problems = ['A', 'B', 'C', 'D', 'E']
scores = [a, b, c, d, e]

participants = []
for mask in range(1, 32):
    total = 0
    name = []
    for i in range(5):
        if (mask >> i) & 1:
            name.append(problems[i])
            total += scores[i]
    participants.append((total, ''.join(name)))

# Sort by descending total, then ascending lex order
participants.sort(key=lambda x: (-x[0], x[1]))

for p in participants:
    print(p[1])