from collections import deque

def solve():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(input().strip())
    
    # Check if a cell is blocked (has adjacent magnet)
    def is_blocked(r, c):
        if grid[r][c] == '#':
            return True
        
        # Check all 4 adjacent cells
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '#':
                return True
        return False
    
    # Find all non-blocked empty cells
    non_blocked = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '.' and not is_blocked(r, c):
                non_blocked.add((r, c))
    
    # Find connected components using BFS
    visited = set()
    max_freedom = 0
    
    for start_r, start_c in non_blocked:
        if (start_r, start_c) in visited:
            continue
            
        # BFS to find all cells reachable from this starting point
        component_size = 0
        queue = deque([(start_r, start_c)])
        component_visited = set()
        component_visited.add((start_r, start_c))
        
        while queue:
            r, c = queue.popleft()
            component_size += 1
            
            # Check all 4 adjacent cells
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W and 
                    (nr, nc) not in component_visited and 
                    (nr, nc) in non_blocked):
                    component_visited.add((nr, nc))
                    queue.append((nr, nc))
        
        # Mark all cells in this component as visited
        visited.update(component_visited)
        max_freedom = max(max_freedom, component_size)
    
    # Handle case where all empty cells are blocked
    if max_freedom == 0:
        # Each blocked empty cell has degree of freedom 1 (itself)
        for r in range(H):
            for c in range(W):
                if grid[r][c] == '.':
                    max_freedom = 1
                    break
            if max_freedom > 0:
                break
    
    print(max_freedom)

solve()