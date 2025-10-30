N = int(input())

grid = [[0 for _ in range(N)] for _ in range(N)]
center = N // 2
grid[center][center] = 'T'

# Directions: right, down, left, up
dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]

x, y = 0, 0
direction = 0  # Start going right
num = 1

for _ in range(N * N - 1):
    grid[x][y] = num
    num += 1
    
    # Calculate next position
    nx, ny = x + dx[direction], y + dy[direction]
    
    # Check if we need to turn (hit boundary or occupied cell)
    if (nx < 0 or nx >= N or ny < 0 or ny >= N or grid[nx][ny] != 0):
        direction = (direction + 1) % 4
        nx, ny = x + dx[direction], y + dy[direction]
    
    x, y = nx, ny

# Print the grid
for i in range(N):
    print(' '.join(str(grid[i][j]) for j in range(N)))