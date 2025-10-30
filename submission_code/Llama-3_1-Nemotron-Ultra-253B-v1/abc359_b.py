n = int(input())
a = list(map(int, input().split()))
positions = {}
for idx, num in enumerate(a, 1):
    if num not in positions:
        positions[num] = []
    positions[num].append(idx)
count = 0
for color in range(1, n + 1):
    pos = positions[color]
    if pos[1] - pos[0] == 2:
        count += 1
print(count)