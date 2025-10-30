# YOUR CODE HERE
H, W, N = map(int, input().split())

# Initialize grid (all white initially)
grid = [[0 for _ in range(W)] for _ in range(H)]

# Starting position and direction
row, col = 0, 0  # (1,1) in 1-indexed becomes (0,0) in 0-indexed
direction = 0  # 0=up, 1=right, 2=down, 3=left

# Direction vectors
dr = [-1, 0, 1, 0]  # row changes for up, right, down, left
dc = [0, 1, 0, -1]  # col changes for up, right, down, left

# Perform N operations
for _ in range(N):
    if grid[row][col] == 0:  # white cell
        grid[row][col] = 1  # paint black
        direction = (direction + 1) % 4  # turn right
    else:  # black cell
        grid[row][col] = 0  # paint white
        direction = (direction - 1) % 4  # turn left
    
    # Move forward
    row = (row + dr[direction]) % H
    col = (col + dc[direction]) % W

# Print the grid
for i in range(H):
    line = ""
    for j in range(W):
        if grid[i][j] == 0:
            line += "."
        else:
            line += "#"
    print(line)