# YOUR CODE HERE
H, W, Q = map(int, input().split())

# Initialize grid with all walls (True means wall exists)
grid = [[True for _ in range(W)] for _ in range(H)]

# Process each query
for _ in range(Q):
    r, c = map(int, input().split())
    r -= 1  # Convert to 0-indexed
    c -= 1
    
    if grid[r][c]:
        # If there's a wall at the bomb position, destroy it
        grid[r][c] = False
    else:
        # If no wall at bomb position, destroy nearest walls in 4 directions
        
        # Look up (decreasing row)
        for i in range(r - 1, -1, -1):
            if grid[i][c]:
                grid[i][c] = False
                break
        
        # Look down (increasing row)
        for i in range(r + 1, H):
            if grid[i][c]:
                grid[i][c] = False
                break
        
        # Look left (decreasing column)
        for j in range(c - 1, -1, -1):
            if grid[r][j]:
                grid[r][j] = False
                break
        
        # Look right (increasing column)
        for j in range(c + 1, W):
            if grid[r][j]:
                grid[r][j] = False
                break

# Count remaining walls
remaining_walls = sum(sum(row) for row in grid)
print(remaining_walls)