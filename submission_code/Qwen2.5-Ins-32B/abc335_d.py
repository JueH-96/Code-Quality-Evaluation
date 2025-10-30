import sys

def solve(N):
    grid = [[0] * N for _ in range(N)]
    center = N // 2
    grid[center][center] = 'T'
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    x, y = center, center - 1
    num = 1
    for i in range(N * N - 1):
        grid[x][y] = num
        num += 1
        dx, dy = directions[i % 4]
        nx, ny = x + dx, y + dy
        if nx < 0 or nx >= N or ny < 0 or ny >= N or grid[nx][ny] != 0:
            directions = directions[1:] + directions[:1]
            dx, dy = directions[0]
        x, y = x + dx, y + dy
    for row in grid:
        print(' '.join(str(cell) if cell != 'T' else 'T' for cell in row))

if __name__ == "__main__":
    N = int(input().strip())
    solve(N)