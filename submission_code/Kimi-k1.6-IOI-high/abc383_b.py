import itertools

h, w, d = map(int, input().split())
floors = []
for i in range(h):
    line = input().strip()
    for j in range(w):
        if line[j] == '.':
            floors.append((i+1, j+1))

max_count = 0
for a, b in itertools.combinations(floors, 2):
    count = 0
    for cell in floors:
        da = abs(cell[0] - a[0]) + abs(cell[1] - a[1])
        db = abs(cell[0] - b[0]) + abs(cell[1] - b[1])
        if da <= d or db <= d:
            count += 1
    if count > max_count:
        max_count = count

print(max_count)