grid = []
for _ in range(8):
    grid.append(input())

rows = [False] * 8
cols = [False] * 8

for i in range(8):
    for j in range(8):
        if grid[i][j] == '#':
            rows[i] = True
            cols[j] = True

count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.' and not rows[i] and not cols[j]:
            count += 1

print(count)