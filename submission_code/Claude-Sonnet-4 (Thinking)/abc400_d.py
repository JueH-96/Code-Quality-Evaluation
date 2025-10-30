from collections import deque

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
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(r, c):
        return 0 <= r < H and 0 <= c < W
    
    def get_reachable_positions(grid, start_r, start_c):
        """Get all positions reachable from start using only movement"""
        visited = set()
        queue = deque([(start_r, start_c)])
        visited.add((start_r, start_c))
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc) and grid[nr][nc] == '.' and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return visited
    
    def apply_front_kick(grid, r, c, direction):
        """Apply front kick from (r, c) in the given direction"""
        new_grid = [row[:] for row in grid]  # Deep copy
        dr, dc = direction
        
        for step in range(1, 3):  # 1 step and 2 steps away
            nr, nc = r + step * dr, c + step * dc
            if is_valid(nr, nc):
                new_grid[nr][nc] = '.'  # Turn into road
        
        return new_grid
    
    def grid_to_tuple(grid):
        return tuple(tuple(row) for row in grid)
    
    # Check if already reachable
    initial_reachable = get_reachable_positions(grid, A, B)
    if (C, D) in initial_reachable:
        return 0
    
    # BFS on the number of front kicks
    queue = deque([(0, grid)])  # (front_kicks, grid)
    visited_grids = set()
    visited_grids.add(grid_to_tuple(grid))
    
    while queue:
        front_kicks, current_grid = queue.popleft()
        
        # Get all reachable positions from start
        reachable_positions = get_reachable_positions(current_grid, A, B)
        
        # Try front kick from each reachable position in each direction
        for r, c in reachable_positions:
            for direction in directions:
                new_grid = apply_front_kick(current_grid, r, c, direction)
                
                # Check if target is now reachable
                new_reachable = get_reachable_positions(new_grid, A, B)
                if (C, D) in new_reachable:
                    return front_kicks + 1
                
                # Add to queue if not visited
                new_grid_tuple = grid_to_tuple(new_grid)
                if new_grid_tuple not in visited_grids:
                    visited_grids.add(new_grid_tuple)
                    queue.append((front_kicks + 1, new_grid))
    
    return -1  # Should not happen

print(solve())