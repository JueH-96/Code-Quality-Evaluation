from collections import deque

def read_grid(H, W):
    grid = []
    for _ in range(H):
        row = list(map(int, input().split()))
        grid.append(row)
    return grid

def grid_to_tuple(grid):
    return tuple(tuple(row) for row in grid)

def tuple_to_grid(t):
    return [list(row) for row in t]

def swap_rows(grid, i):
    new_grid = [row[:] for row in grid]
    new_grid[i], new_grid[i+1] = new_grid[i+1], new_grid[i]
    return new_grid

def swap_cols(grid, j):
    new_grid = [row[:] for row in grid]
    for i in range(len(grid)):
        new_grid[i][j], new_grid[i][j+1] = new_grid[i][j+1], new_grid[i][j]
    return new_grid

def can_transform(A, B):
    # Check if both grids contain the same elements
    elements_A = []
    elements_B = []
    
    for row in A:
        elements_A.extend(row)
    for row in B:
        elements_B.extend(row)
    
    elements_A.sort()
    elements_B.sort()
    
    return elements_A == elements_B

def min_operations(A, B):
    H = len(A)
    W = len(A[0])
    
    if not can_transform(A, B):
        return -1
    
    start = grid_to_tuple(A)
    target = grid_to_tuple(B)
    
    if start == target:
        return 0
    
    visited = {start}
    queue = deque([(start, 0)])
    
    while queue:
        current_tuple, ops = queue.popleft()
        current = tuple_to_grid(current_tuple)
        
        # Try swapping adjacent rows
        for i in range(H - 1):
            new_grid = swap_rows(current, i)
            new_tuple = grid_to_tuple(new_grid)
            
            if new_tuple == target:
                return ops + 1
            
            if new_tuple not in visited:
                visited.add(new_tuple)
                queue.append((new_tuple, ops + 1))
        
        # Try swapping adjacent columns
        for j in range(W - 1):
            new_grid = swap_cols(current, j)
            new_tuple = grid_to_tuple(new_grid)
            
            if new_tuple == target:
                return ops + 1
            
            if new_tuple not in visited:
                visited.add(new_tuple)
                queue.append((new_tuple, ops + 1))
    
    return -1

# Read input
H, W = map(int, input().split())
A = read_grid(H, W)
B = read_grid(H, W)

# Solve and output
result = min_operations(A, B)
print(result)