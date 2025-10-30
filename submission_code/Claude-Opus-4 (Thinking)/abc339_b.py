# YOUR CODE HERE
H, W, N = map(int, input().split())

# Initialize grid as all white (False = white, True = black)
grid = [[False for _ in range(W)] for _ in range(H)]

# Starting position and direction (0-indexed)
row, col = 0, 0
# Direction: 0=up, 1=right, 2=down, 3=left
direction = 0

# Direction vectors (up, right, down, left)
dr = [-1, 0, 1, 0]
dc = [0, 1, 0, -1]

# Perform N operations
for _ in range(N):
    if grid[row][col]:  # Black cell
        # Paint white
        grid[row][col] = False
        # Rotate counterclockwise
        direction = (direction - 1) % 4
    else:  # White cell
        # Paint black
        grid[row][col] = True
        # Rotate clockwise
        direction = (direction + 1) % 4
    
    # Move forward
    row = (row + dr[direction]) % H
    col = (col + dc[direction]) % W

# Print the grid
for i in range(H):
    print(''.join('#' if grid[i][j] else '.' for j in range(W)))