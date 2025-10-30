n = int(input())
grid = [list(input().strip()) for _ in range(n)]

result = [row.copy() for row in grid]

for a in range(n):
    for b in range(n):
        k = min(a + 1, n - a, b + 1, n - b)
        if k % 2 == 1:
            x = n - b
            y = a - 1
            if 0 <= x < n and 0 <= y < n:
                result[a][b] = grid[x][y]
            else:
                result[a][b] = grid[a][b]
        else:
            result[a][b] = grid[a][b]

for row in result:
    print(''.join(row))