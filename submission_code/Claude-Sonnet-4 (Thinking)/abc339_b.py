H, W, N = map(int, input().split())

grid = [[False] * W for _ in range(H)]  # False = white, True = black
directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
direction = 0  # Start facing up
row, col = 0, 0  # Start at (0,0)

for _ in range(N):
    if grid[row][col]:  # Current cell is black
        grid[row][col] = False  # Paint white
        direction = (direction - 1) % 4  # Rotate counterclockwise
    else:  # Current cell is white
        grid[row][col] = True  # Paint black
        direction = (direction + 1) % 4  # Rotate clockwise
    
    # Move forward in current direction
    dr, dc = directions[direction]
    row = (row + dr) % H
    col = (col + dc) % W

# Print the grid
for r in range(H):
    print(''.join('#' if grid[r][c] else '.' for c in range(W)))