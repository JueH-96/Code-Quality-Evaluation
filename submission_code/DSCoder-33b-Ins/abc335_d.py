# YOUR CODE HERE

import sys

def solve(N):
    grid = [[0]*N for _ in range(N)]
    x, y = N//2, N//2
    dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
    dir = 0
    for i in range(1, N*N):
        grid[x][y] = i
        nx, ny = x + dx[dir], y + dy[dir]
        if nx < 0 or nx >= N or ny < 0 or ny >= N or grid[nx][ny] != 0:
            dir = (dir + 1) % 4
            nx, ny = x + dx[dir], y + dy[dir]
        x, y = nx, ny
    grid[N//2][N//2] = 'T'
    return grid

N = int(sys.stdin.readline().strip())
grid = solve(N)
for row in grid:
    print(' '.join(map(str, row)))