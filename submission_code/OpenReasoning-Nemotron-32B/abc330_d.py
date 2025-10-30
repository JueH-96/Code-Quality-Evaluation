n = int(input().strip())
grid = [input().strip() for _ in range(n)]

row_count = [0] * n
col_count = [0] * n

for i in range(n):
	for j in range(n):
		if grid[i][j] == 'o':
			row_count[i] += 1
			col_count[j] += 1

ans = 0
for i in range(n):
	for j in range(n):
		if grid[i][j] == 'o':
			ans += (row_count[i] - 1) * (col_count[j] - 1)

print(ans)