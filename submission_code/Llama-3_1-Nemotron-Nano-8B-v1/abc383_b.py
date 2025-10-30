import itertools

h, w, d = map(int, input().split())
grid = [input().strip() for _ in range(h)]

floor_cells = []
for i in range(h):
    for j in range(w):
        if grid[i][j] == '.':
            floor_cells.append((i, j))

max_humid = 0

for a, b in itertools.combinations(floor_cells, 2):
    count = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                distance_a = abs(i - a[0]) + abs(j - a[1])
                distance_b = abs(i - b[0]) + abs(j - b[1])
                if distance_a <= d or distance_b <= d:
                    count += 1
    max_humid = max(max_humid, count)

print(max_humid)