scores = list(map(int, input().split()))
participants = []

for mask in range(1, 32):
    name = []
    total = 0
    for i in range(5):
        if mask & (1 << i):
            name.append(chr(ord('A') + i))
            total += scores[i]
    participants.append((''.join(name), total))

participants.sort(key=lambda x: (-x[1], x[0]))

for p in participants:
    print(p[0])