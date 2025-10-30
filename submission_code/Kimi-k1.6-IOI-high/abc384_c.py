a, b, c, d, e = map(int, input().split())
values = [a, b, c, d, e]

participants = []
for mask in range(1, 32):
    name = ''.join([chr(ord('A') + i) for i in range(5) if (mask & (1 << i))])
    score = sum(values[i] for i in range(5) if (mask & (1 << i)))
    participants.append((-score, name))  # Using negative score for reverse sort

# Sort by score descending (hence the negative), then by name ascending
participants.sort()
for p in participants:
    print(p[1])