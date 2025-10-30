# YOUR CODE HERE
import sys
from collections import deque

def solve():
    """
    Reads the grid, calculates the maximum degree of freedom, and prints the result.
    """
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Step 1: Pre-calculate which empty cells are "movable".
    # A cell is movable if it's empty ('.') and has no adjacent magnets ('#').
    is_movable = [[False] * W for _ in range(H)]
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    for r in range(H):
        for c in range(W):
            if grid[r][c] == '#':
                continue
            
            has_magnet_neighbor = False
            for i in range(4):
                nr, nc = r + dr[i], c + dc[i]
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == '#':
                    has_magnet_neighbor = True
                    break
            
            if not has_magnet_neighbor:
                is_movable[r][c] = True

    # Step 2: Find connected components of movable cells and calculate max freedom.
    # The degree of freedom is at least 1.
    max_freedom = 1
    visited = [[False] * W for _ in range(H)]

    for r_start in range(H):
        for c_start in range(W):
            # If we find an unvisited movable cell, it's the start of a new component.
            if is_movable[r_start][c_start] and not visited[r_start][c_start]:
                
                # Start a BFS to find the size of the movable component and its neighbors.
                q = deque([(r_start, c_start)])
                visited[r_start][c_start] = True
                
                movable_component_size = 0
                immovable_neighbors = set()

                while q:
                    curr_r, curr_c = q.popleft()
                    movable_component_size += 1
                    
                    # Check all four neighbors.
                    for i in range(4):
                        nr, nc = curr_r + dr[i], curr_c + dc[i]
                        
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue

                        # If the neighbor is also movable and unvisited, add it to the component.
                        if is_movable[nr][nc] and not visited[nr][nc]:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                        # If the neighbor is an empty but "immovable" cell, it's a reachable dead-end.
                        elif grid[nr][nc] == '.' and not is_movable[nr][nc]:
                            immovable_neighbors.add((nr, nc))

                # The freedom for this component is the sum of its movable cells and adjacent immovable cells.
                current_freedom = movable_component_size + len(immovable_neighbors)
                max_freedom = max(max_freedom, current_freedom)

    print(max_freedom)

solve()