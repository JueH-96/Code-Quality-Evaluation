H, W, N = map(int, input().split())

# Initialize the grid (all white)
grid = [['.' for _ in range(W+1)] for _ in range(H+1)]

# Initial position and direction
row, col = 1, 1
direction = 0  # 0: up, 1: right, 2: down, 3: left

# Define movement for each direction
dr = [-1, 0, 1, 0]
dc = [0, 1, 0, -1]

for _ in range(N):
    # Check the color of the current cell
    if grid[row][col] == '.':  # If it's white
        grid[row][col] = '#'  # Paint it black
        direction = (direction + 1) % 4  # Rotate 90° clockwise
    else:  # If it's black
        grid[row][col] = '.'  # Paint it white
        direction = (direction - 1) % 4  # Rotate 90° counterclockwise
    
    # Move forward in the new direction
    row += dr[direction]
    col += dc[direction]
    
    # Handle toroidal wrapping
    if row == 0:
        row = H
    elif row == H+1:
        row = 1
    
    if col == 0:
        col = W
    elif col == W+1:
        col = 1

# Print the grid
for i in range(1, H+1):
    print(''.join(grid[i][1:W+1]))