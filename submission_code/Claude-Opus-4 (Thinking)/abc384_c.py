# YOUR CODE HERE
a, b, c, d, e = map(int, input().split())
problems = "ABCDE"
scores = [a, b, c, d, e]

participants = []
for mask in range(1, 32):  # 1 to 31
    name = ""
    total_score = 0
    for i in range(5):
        if mask & (1 << i):
            name += problems[i]
            total_score += scores[i]
    participants.append((name, total_score))

# Sort by score (descending) and then by name (ascending)
participants.sort(key=lambda x: (-x[1], x[0]))

for name, _ in participants:
    print(name)