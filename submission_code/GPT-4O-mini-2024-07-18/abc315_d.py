def remaining_cookies(H, W, grid):
    while True:
        marked = set()
        
        # Mark rows
        for i in range(H):
            count = {}
            for j in range(W):
                if grid[i][j] not in count:
                    count[grid[i][j]] = 0
                count[grid[i][j]] += 1
            
            for j in range(W):
                if count[grid[i][j]] >= 2:
                    marked.add((i, j))
        
        # Mark columns
        for j in range(W):
            count = {}
            for i in range(H):
                if grid[i][j] not in count:
                    count[grid[i][j]] = 0
                count[grid[i][j]] += 1
            
            for i in range(H):
                if count[grid[i][j]] >= 2:
                    marked.add((i, j))
        
        # If no cookies are marked, we are done
        if not marked:
            break
        
        # Remove marked cookies
        for i, j in marked:
            grid[i][j] = None  # Mark as removed
        
        # Clean up the grid
        grid = [[grid[i][j] for j in range(W) if grid[i][j] is not None] for i in range(H)]
        grid = [row for row in grid if row]  # Remove empty rows
        
        # Update H and W
        H = len(grid)
        W = len(grid[0]) if H > 0 else 0
    
    # Count remaining cookies
    remaining_count = sum(len(row) for row in grid)
    return remaining_count

import sys
input = sys.stdin.read
data = input().splitlines()

H, W = map(int, data[0].split())
grid = [list(data[i + 1]) for i in range(H)]

result = remaining_cookies(H, W, grid)
print(result)