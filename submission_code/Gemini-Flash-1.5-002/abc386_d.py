n, m = map(int, input().split())
grid = {}
for i in range(m):
    x, y, c = input().split()
    x, y = int(x), int(y)
    grid[(x, y)] = c

min_row = {}
max_row = {}
min_col = {}
max_col = {}

for x, y in grid:
    if grid[(x, y)] == 'B':
        if x not in min_row or y < min_row[x]:
            min_row[x] = y
        if x not in max_row or y > max_row[x]:
            max_row[x] = y
        if y not in min_col or x < min_col[y]:
            min_col[y] = x
        if y not in max_col or x > max_col[y]:
            max_col[y] = x

possible = True
for x in min_row:
    for y in range(1, min_row[x]):
        if (x,y) in grid and grid[(x,y)] == 'W':
            possible = False
            break
    if not possible:
        break

for x in max_row:
    for y in range(max_row[x] + 1, n + 1):
        if (x,y) in grid and grid[(x,y)] == 'B':
            possible = False
            break
    if not possible:
        break

for y in min_col:
    for x in range(1, min_col[y]):
        if (x,y) in grid and grid[(x,y)] == 'W':
            possible = False
            break
    if not possible:
        break

for y in max_col:
    for x in range(max_col[y] + 1, n + 1):
        if (x,y) in grid and grid[(x,y)] == 'B':
            possible = False
            break
    if not possible:
        break

if possible:
    print("Yes")
else:
    print("No")