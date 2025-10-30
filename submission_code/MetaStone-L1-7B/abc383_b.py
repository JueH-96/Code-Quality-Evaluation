H, W, D = map(int, input().split())
grid = []
floors = []

for i in range(H):
    line = input().strip()
    grid.append(line)
    for j in range(W):
        if line[j] == '.':
            floors.append((i, j))

max_count = 0

for i in range(len(floors)):
    for j in range(i + 1, len(floors)):
        a = floors[i]
        b = floors[j]
        covered = set()
        for x in range(H):
            for y in range(W):
                if grid[x][y] == '.':
                    d1 = abs(x - a[0]) + abs(y - a[1])
                    d2 = abs(x - b[0]) + abs(y - b[1])
                    if d1 <= D or d2 <= D:
                        covered.add((x, y))
        count = len(covered)
        if count > max_count:
            max_count = count

print(max_count)