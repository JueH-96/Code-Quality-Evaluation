scores = list(map(int, input().split()))
participants = []
for mask in range(1, 1 << 5):
    name = []
    total = 0
    for i in range(5):
        if mask & (1 << i):
            name.append('ABCDE'[i])
            total += scores[i]
    participants.append((-total, ''.join(name)))
participants.sort()
for p in participants:
    print(p[1])