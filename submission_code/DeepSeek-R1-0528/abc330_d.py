n = int(input().strip())
grid = []
for i in range(n):
    grid.append(input().strip())

row_count = [0] * n
col_count = [0] * n

for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            row_count[i] += 1
            col_count[j] += 1

total = 0
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            total += (row_count[i] - 1) * (col_count[j] - 1)

print(total)