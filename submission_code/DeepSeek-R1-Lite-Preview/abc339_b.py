H, W, N = map(int, input().split())

# Initialize grid with all white (0)
grid = [[0 for _ in range(W)] for _ in range(H)]

# Define directions: up, right, down, left
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
current_dir = 0  # start facing up
row, col = 0, 0  # starting position (0,0)

for _ in range(N):
    if grid[row][col] == 0:
        # Repaint to black
        grid[row][col] = 1
        # Rotate 90 degrees clockwise
        current_dir = (current_dir + 1) % 4
    else:
        # Repaint to white
        grid[row][col] = 0
        # Rotate 90 degrees counterclockwise
        current_dir = (current_dir - 1) % 4
    # Move forward in the new direction
    row = (row + directions[current_dir][0]) % H
    col = (col + directions[current_dir][1]) % W

# Print the grid
for r in range(H):
    line = ''.join(['.' if cell == 0 else '#' for cell in grid[r]])
    print(line)