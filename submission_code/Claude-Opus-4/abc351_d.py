# YOUR CODE HERE
from collections import deque

def can_move(grid, i, j, H, W):
    """Check if a cell can move (no magnets in adjacent cells)"""
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
            return False
    return True

def bfs(grid, start_i, start_j, H, W, can_move_cells):
    """Find all reachable cells from a starting cell"""
    visited = set()
    queue = deque([(start_i, start_j)])
    visited.add((start_i, start_j))
    
    while queue:
        i, j = queue.popleft()
        
        # Try to move to adjacent cells
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (0 <= ni < H and 0 <= nj < W and 
                (ni, nj) not in visited and 
                grid[ni][nj] == '.' and
                (ni, nj) in can_move_cells):
                visited.add((ni, nj))
                queue.append((ni, nj))
    
    return len(visited)

# Read input
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Find all cells that can move
can_move_cells = set()
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.' and can_move(grid, i, j, H, W):
            can_move_cells.add((i, j))

# Find maximum degree of freedom
max_freedom = 0

# For each cell without a magnet
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':
            if (i, j) in can_move_cells:
                # This cell can move, so find its connected component
                freedom = bfs(grid, i, j, H, W, can_move_cells)
            else:
                # This cell cannot move, so its degree of freedom is 1
                freedom = 1
            max_freedom = max(max_freedom, freedom)

print(max_freedom)