def solve(N):
    grid = [[0]*N for _ in range(N)]
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    x, y, d = N//2, N//2, 0
    for i in range(1, N*N+1):
        grid[x][y] = i
        if i == N*N:
            grid[x][y] = 'T'
        nx, ny = x+dx[d], y+dy[d]
        if 0 <= nx < N and 0 <= ny < N and grid[nx][ny] == 0:
            x, y = nx, ny
        else:
            d = (d+1)%4
            x, y = x+dx[d], y+dy[d]
    return grid

N = int(input())
grid = solve(N)
for row in grid:
    print(*row)