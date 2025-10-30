n = int(input().strip())
grid = []
for _ in range(n):
    grid.append(input().strip())

result = [[' '] * n for _ in range(n)]

for i in range(n):
    for j in range(n):
        d = min(i, j, n - 1 - i, n - 1 - j)
        r = (d + 1) % 4
        if r == 0:
            ni, nj = i, j
        elif r == 1:
            ni, nj = j, n - 1 - i
        elif r == 2:
            ni, nj = n - 1 - i, n - 1 - j
        else:
            ni, nj = n - 1 - j, i
        result[ni][nj] = grid[i][j]

for row in result:
    print(''.join(row))