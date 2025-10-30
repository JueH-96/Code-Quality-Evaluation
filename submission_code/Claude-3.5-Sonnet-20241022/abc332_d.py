from itertools import permutations
from collections import deque

def get_grid():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(map(int, input().split())))
    return grid

def grid_to_tuple(grid):
    return tuple(tuple(row) for row in grid)

def swap_rows(grid, i):
    grid = [list(row) for row in grid]
    grid[i], grid[i+1] = grid[i+1], grid[i]
    return grid_to_tuple(grid)

def swap_cols(grid, j):
    grid = [list(row) for row in grid]
    for i in range(len(grid)):
        grid[i][j], grid[i][j+1] = grid[i][j+1], grid[i][j]
    return grid_to_tuple(grid)

def solve():
    H, W = map(int, input().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, input().split())))
    B = []
    for _ in range(H):
        B.append(list(map(int, input().split())))
        
    target = grid_to_tuple(B)
    start = grid_to_tuple(A)
    
    if start == target:
        return 0
        
    seen = {start}
    q = deque([(start, 0)])
    
    while q:
        curr_grid, steps = q.popleft()
        
        # Try row swaps
        for i in range(H-1):
            next_grid = swap_rows(curr_grid, i)
            if next_grid == target:
                return steps + 1
            if next_grid not in seen:
                seen.add(next_grid)
                q.append((next_grid, steps + 1))
                
        # Try column swaps
        for j in range(W-1):
            next_grid = swap_cols(curr_grid, j)
            if next_grid == target:
                return steps + 1
            if next_grid not in seen:
                seen.add(next_grid)
                q.append((next_grid, steps + 1))
                
    return -1

print(solve())