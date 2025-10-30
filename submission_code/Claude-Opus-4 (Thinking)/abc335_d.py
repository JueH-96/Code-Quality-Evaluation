n = int(input())

grid = [[0] * n for _ in range(n)]
center = n // 2

# Directions: right, down, left, up
directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
dir_idx = 0

x, y = 0, 0
num = 1

for _ in range(n * n - 1):
    grid[x][y] = num
    num += 1
    
    # Find the next position
    for _ in range(4):  # Try all four directions
        dx, dy = directions[dir_idx]
        nx, ny = x + dx, y + dy
        
        # Check if the next position is valid
        if (0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 0 and 
            (nx != center or ny != center)):
            x, y = nx, ny
            break
        else:
            # Turn clockwise
            dir_idx = (dir_idx + 1) % 4

# Place Takahashi at the center
grid[center][center] = 'T'

# Print the grid
for row in grid:
    print(' '.join(str(cell) for cell in row))