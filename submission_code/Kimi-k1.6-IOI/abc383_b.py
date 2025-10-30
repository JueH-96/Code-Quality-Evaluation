import itertools

h, w, d = map(int, input().split())
grid = []
floors = []
for i in range(h):
    line = input().strip()
    grid.append(line)
    for j in range(w):
        if line[j] == '.':
            floors.append((i, j))

max_count = 0

for a, b in itertools.combinations(floors, 2):
    count = 0
    for x, y in floors:
        dist_a = abs(x - a[0]) + abs(y - a[1])
        dist_b = abs(x - b[0]) + abs(y - b[1])
        if dist_a <= d or dist_b <= d:
            count += 1
    if count > max_count:
        max_count = count

print(max_count)