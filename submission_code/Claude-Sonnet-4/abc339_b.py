H, W, N = map(int, input().split())

# Initialize grid (False = white, True = black)
grid = [[False for _ in range(W)] for _ in range(H)]

# Starting position (0-indexed) and direction
row, col = 0, 0
direction = 0  # 0=up, 1=right, 2=down, 3=left

# Direction vectors: up, right, down, left
dr = [-1, 0, 1, 0]
dc = [0, 1, 0, -1]

for _ in range(N):
    if not grid[row][col]:  # Current cell is white
        grid[row][col] = True  # Paint black
        direction = (direction + 1) % 4  # Rotate clockwise
    else:  # Current cell is black
        grid[row][col] = False  # Paint white
        direction = (direction - 1) % 4  # Rotate counterclockwise
    
    # Move forward one cell in current direction
    row = (row + dr[direction]) % H
    col = (col + dc[direction]) % W

# Output the grid
for i in range(H):
    line = ""
    for j in range(W):
        if grid[i][j]:
            line += "#"
        else:
            line += "."
    print(line)