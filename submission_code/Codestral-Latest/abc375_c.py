import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
grid = [list(data[i]) for i in range(1, N + 1)]

for i in range(1, N // 2 + 1):
    for x in range(i, N + 1 - i):
        for y in range(i, N + 1 - i):
            grid[y - 1][N - x - 1], grid[x - 1][y - 1] = grid[x - 1][y - 1], grid[y - 1][N - x - 1]

for row in grid:
    print(''.join(row))