import itertools

h, w, D = map(int, input().split())
grid = [input().strip() for _ in range(h)]

floors = []
for i in range(h):
    for j in range(w):
        if grid[i][j] == '.':
            floors.append((i, j))

max_count = 0
for a, b in itertools.combinations(floors, 2):
    count = 0
    for (ci, cj) in floors:
        d1 = abs(ci - a[0]) + abs(cj - a[1])
        d2 = abs(ci - b[0]) + abs(cj - b[1])
        if d1 <= D or d2 <= D:
            count += 1
    if count > max_count:
        max_count = count

print(max_count)