from collections import deque
import copy

def solve():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(input().strip()))
    
    A, B, C, D = map(int, input().split())
    A -= 1  # Convert to 0-indexed
    B -= 1
    C -= 1
    D -= 1
    
    # BFS with state = (row, col, grid_state, kicks)
    # Use string representation of grid as state key
    def grid_to_string(g):
        return ''.join(''.join(row) for row in g)
    
    queue = deque([(A, B, grid, 0)])
    visited = set()
    visited.add((A, B, grid_to_string(grid)))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    while queue:
        row, col, current_grid, kicks = queue.popleft()
        
        # Check if reached destination
        if row == C and col == D:
            return kicks
        
        # Try moving to adjacent cells
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < H and 0 <= new_col < W and current_grid[new_row][new_col] == '.':
                state_key = (new_row, new_col, grid_to_string(current_grid))
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((new_row, new_col, current_grid, kicks))
        
        # Try front kicks in 4 directions
        for dr, dc in directions:
            new_grid = [row[:] for row in current_grid]  # Deep copy
            changed = False
            
            # Check cells 1 and 2 steps away
            for steps in [1, 2]:
                kick_row = row + dr * steps
                kick_col = col + dc * steps
                if 0 <= kick_row < H and 0 <= kick_col < W:
                    if new_grid[kick_row][kick_col] == '#':
                        new_grid[kick_row][kick_col] = '.'
                        changed = True
            
            if changed:
                state_key = (row, col, grid_to_string(new_grid))
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((row, col, new_grid, kicks + 1))
    
    return -1  # Should not reach here given problem constraints

print(solve())