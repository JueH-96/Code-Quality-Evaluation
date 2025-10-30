# YOUR CODE HERE
H, W, Q = map(int, input().split())

# Initialize grid with all walls (True = wall exists)
grid = [[True] * W for _ in range(H)]

for _ in range(Q):
    R, C = map(int, input().split())
    # Convert to 0-indexed
    R -= 1
    C -= 1
    
    if grid[R][C]:
        # Wall exists at bomb position, destroy it
        grid[R][C] = False
    else:
        # No wall at bomb position, destroy first walls in 4 directions
        
        # Look up
        for i in range(R - 1, -1, -1):
            if grid[i][C]:
                grid[i][C] = False
                break
        
        # Look down
        for i in range(R + 1, H):
            if grid[i][C]:
                grid[i][C] = False
                break
        
        # Look left
        for j in range(C - 1, -1, -1):
            if grid[R][j]:
                grid[R][j] = False
                break
        
        # Look right
        for j in range(C + 1, W):
            if grid[R][j]:
                grid[R][j] = False
                break

# Count remaining walls
count = sum(sum(row) for row in grid)
print(count)