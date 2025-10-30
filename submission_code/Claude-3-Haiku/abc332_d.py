from collections import deque

def solve():
    H, W = map(int, input().split())
    
    A = []
    for _ in range(H):
        A.append(list(map(int, input().split())))
    
    B = []
    for _ in range(H):
        B.append(list(map(int, input().split())))
    
    # Check if grids are already identical
    if A == B:
        return 0
    
    # Create a queue to store the current grid and the number of operations
    queue = deque([(A, 0)])
    
    while queue:
        grid, ops = queue.popleft()
        
        # Check all possible row swaps
        for i in range(H-1):
            new_grid = [row[:] for row in grid]
            new_grid[i], new_grid[i+1] = new_grid[i+1], new_grid[i]
            if new_grid == B:
                return ops + 1
            queue.append((new_grid, ops + 1))
        
        # Check all possible column swaps
        for j in range(W-1):
            new_grid = [row[:] for row in grid]
            for i in range(H):
                new_grid[i][j], new_grid[i][j+1] = new_grid[i][j+1], new_grid[i][j]
            if new_grid == B:
                return ops + 1
            queue.append((new_grid, ops + 1))
    
    # If no solution is found, return -1
    return -1