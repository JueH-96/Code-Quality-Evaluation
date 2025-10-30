H, W, Q = map(int, input().split())

# Initialize grid with walls in every cell (True means a wall exists)
grid = [[True for _ in range(W)] for _ in range(H)]

for _ in range(Q):
    R, C = map(int, input().split())
    R -= 1  # Convert to 0-based indexing
    C -= 1
    
    if grid[R][C]:
        # There is a wall at (R, C), destroy it and end the process
        grid[R][C] = False
    else:
        # No wall at (R, C), look in four directions
        
        # Look up
        for i in range(R-1, -1, -1):
            if grid[i][C]:
                grid[i][C] = False
                break
        
        # Look down
        for i in range(R+1, H):
            if grid[i][C]:
                grid[i][C] = False
                break
        
        # Look left
        for j in range(C-1, -1, -1):
            if grid[R][j]:
                grid[R][j] = False
                break
        
        # Look right
        for j in range(C+1, W):
            if grid[R][j]:
                grid[R][j] = False
                break

# Count remaining walls
count = sum(sum(row) for row in grid)
print(count)