n = int(input())
grid = [input().strip() for _ in range(n)]
result = [[None] * n for _ in range(n)]

for x in range(1, n+1):
    for y in range(1, n+1):
        k = min(min(x, n+1 - x), min(y, n+1 - y))
        x0, y0 = x, y
        for _ in range(k):
            x0, y0 = y0, n+1 - x0
        result[x-1][y-1] = grid[x0-1][y0-1]

for row in result:
    print(''.join(row))