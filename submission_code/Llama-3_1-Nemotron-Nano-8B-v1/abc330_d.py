n = int(input())
grid = [input().strip() for _ in range(n)]

row_counts = [row.count('o') for row in grid]
col_counts = []
for j in range(n):
    cnt = 0
    for i in range(n):
        if grid[i][j] == 'o':
            cnt += 1
    col_counts.append(cnt)

ans = 0
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            ans += (row_counts[i] - 1) * (col_counts[j] - 1)

print(ans)