H, W, D = map(int, input().split())
grid = [input().strip() for _ in range(H)]

floors = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            floors.append((i, j))

n = len(floors)
bitmask = [0] * n

for i in range(n):
    x1, y1 = floors[i]
    for j in range(n):
        x2, y2 = floors[j]
        distance = abs(x1 - x2) + abs(y1 - y2)
        if distance <= D:
            bitmask[i] |= 1 << j

max_count = 0
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        combined = bitmask[i] | bitmask[j]
        count = bin(combined).count('1')
        if count > max_count:
            max_count = count

print(max_count)