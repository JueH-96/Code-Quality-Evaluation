grid = [input().strip() for _ in range(8)]

row_safe = [False] * 8
for i in range(8):
    row_safe[i] = '#' not in grid[i]

col_safe = [True] * 8
for j in range(8):
    for i in range(8):
        if grid[i][j] == '#':
            col_safe[j] = False
            break

count = 0
for i in range(8):
    for j in range(8):
        if grid[i][j] == '.' and row_safe[i] and col_safe[j]:
            count += 1

print(count)