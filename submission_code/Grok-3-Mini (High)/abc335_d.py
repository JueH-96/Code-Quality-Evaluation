import sys
data = sys.stdin.read().split()
N = int(data[0])
grid = [[0 for _ in range(N)] for _ in range(N)]
dx = [0, 1, 0, -1]  # right, down, left, up
dy = [1, 0, -1, 0]
x, y = 0, 0
direction = 0
for num in range(1, N*N + 1):
    grid[x][y] = num
    nx = x + dx[direction]
    ny = y + dy[direction]
    if 0 <= nx < N and 0 <= ny < N and grid[nx][ny] == 0:
        x, y = nx, ny
    else:
        direction = (direction + 1) % 4
        x += dx[direction]
        y += dy[direction]
# Now output
center_i = N // 2
center_j = N // 2
for i in range(N):
    for j in range(N):
        if i == center_i and j == center_j:
            val = "T"
        else:
            val = str(grid[i][j])
        if j > 0:
            print(" ", end="")
        print(val, end="")
    print()