from collections import deque

def can_kick_to_road(h, w, grid, r, c, dr, dc):
    # Check if we can kick in direction (dr, dc) from position (r, c)
    # and what cells would be affected
    affected = []
    for dist in range(1, 3):
        nr = r + dr * dist
        nc = c + dc * dist
        if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == '#':
            affected.append((nr, nc))
    return affected

def solve():
    # Read input
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    a, b, c, d = map(int, input().split())
    a -= 1; b -= 1; c -= 1; d -= 1  # Convert to 0-based indexing
    
    # BFS state: (row, col, kicks_used)
    # visited: (row, col, grid_state) -> kicks_used
    visited = {}
    q = deque([(a, b, 0, frozenset())])  # position, kicks, broken_walls
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while q:
        r, c, kicks, broken = q.popleft()
        
        if r == c == -1:  # Invalid state
            continue
            
        if r == c and r == -1:  # Invalid state
            continue
            
        if (r, c) == (c, d):  # Reached destination
            return kicks
            
        state = (r, c, broken)
        if state in visited and visited[state] <= kicks:
            continue
        visited[state] = kicks
        
        # Try moving in all directions
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if grid[nr][nc] == '.' or (nr, nc) in broken:
                    q.append((nr, nc, kicks, broken))
        
        # Try kicking in all directions
        for dr, dc in directions:
            affected = can_kick_to_road(h, w, grid, r, c, dr, dc)
            if affected:
                new_broken = set(broken)
                new_broken.update(affected)
                q.append((r, c, kicks + 1, frozenset(new_broken)))

    return -1  # No solution found

print(solve())