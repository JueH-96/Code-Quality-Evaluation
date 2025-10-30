n = int(input())
grid = [input().strip() for _ in range(n)]

row_counts = [row.count('o') for row in grid]
col_counts = [0] * n
for j in range(n):
    for i in range(n):
        if grid[i][j] == 'o':
            col_counts[j] += 1

ans = 0
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            ans += (row_counts[i] - 1) * (col_counts[j] - 1)

print(ans)