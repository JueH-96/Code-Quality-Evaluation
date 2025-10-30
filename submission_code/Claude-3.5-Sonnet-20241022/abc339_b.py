H, W, N = map(int, input().split())

# Initialize grid
grid = [[False] * W for _ in range(H)]

# Direction vectors (up, right, down, left)
dx = [-1, 0, 1, 0]  
dy = [0, 1, 0, -1]

# Current position and direction
x, y = 0, 0  # (0,0) is (1,1) in problem
dir = 0  # 0:up, 1:right, 2:down, 3:left

for _ in range(N):
    # Change color and direction
    if not grid[x][y]:  # white cell
        grid[x][y] = True  # paint black
        dir = (dir + 1) % 4  # rotate clockwise
    else:  # black cell
        grid[x][y] = False  # paint white
        dir = (dir - 1) % 4  # rotate counterclockwise
    
    # Move forward
    x = (x + dx[dir]) % H
    y = (y + dy[dir]) % W

# Print result
for row in grid:
    print(''.join('#' if cell else '.' for cell in row))