n = int(input())
grid = [input().strip() for _ in range(n)]

# Compute row counts
row_counts = [row.count('o') for row in grid]

# Compute column counts
col_counts = [0] * n
for j in range(n):
    cnt = 0
    for i in range(n):
        if grid[i][j] == 'o':
            cnt += 1
    col_counts[j] = cnt

result = 0
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            result += (row_counts[i] - 1) * (col_counts[j] - 1)

print(result)