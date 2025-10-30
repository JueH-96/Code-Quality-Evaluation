# Read input
H, W, D = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Collect all floor cells
floors = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            floors.append((i, j))

n = len(floors)
max_humidified = 0

# Check all pairs of distinct floors
for i in range(n):
    for j in range(i + 1, n):
        a = floors[i]
        b = floors[j]
        count = 0
        for x in floors:
            dx_a = abs(x[0] - a[0]) + abs(x[1] - a[1])
            dx_b = abs(x[0] - b[0]) + abs(x[1] - b[1])
            if dx_a <= D or dx_b <= D:
                count += 1
        if count > max_humidified:
            max_humidified = count

print(max_humidified)