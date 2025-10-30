import collections

# Read H and W
H, W = map(int, input().split())

# Read the grid
grid = [input() for _ in range(H)]

# Directions for neighbors (up, down, left, right)
dr = [-1, 1, 0, 0]
dc = [0, 0, -1, 1]

# Step 1: Classify each empty cell
# cell_type[r][c] will store one of:
# 'MAGNET': cell contains '#'
# 'BLOCKED': cell contains '.' but has at least one adjacent '#'
# 'MOVABLE': cell contains '.' and has no adjacent '#'
cell_type = [['' for _ in range(W)] for _ in range(H)]

for r in range(H):
    for c in range(W):
        if grid[r][c] == '#':
            cell_type[r][c] = 'MAGNET'
        else: # grid[r][c] == '.'
            is_blocked_by_magnet = False
            for i in range(4): # Check all 4 neighbors
                nr, nc = r + dr[i], c + dc[i]
                # Check if neighbor is within grid boundaries
                if 0 <= nr < H and 0 <= nc < W:
                    if grid[nr][nc] == '#':
                        is_blocked_by_magnet = True
                        break # Found a magnet neighbor, this cell is blocked
            
            if is_blocked_by_magnet:
                cell_type[r][c] = 'BLOCKED'
            else:
                cell_type[r][c] = 'MOVABLE'

# Step 2: Find connected components of 'MOVABLE' cells using BFS
# and calculate the total reachable cells (degree of freedom) for each component.
# Keep track of the maximum degree of freedom found.
visited = [[False for _ in range(W)] for _ in range(H)]
max_degree_of_freedom = 1 # Initialize with 1, as any non-magnet cell always includes itself.

for r in range(H):
    for c in range(W):
        # Only start BFS from unvisited 'MOVABLE' cells.
        # 'BLOCKED' cells have a DOF of 1, which is covered by initialization.
        # 'MAGNET' cells are not considered for DOF.
        if cell_type[r][c] == 'MOVABLE' and not visited[r][c]:
            q = collections.deque([(r, c)])
            visited[r][c] = True
            
            current_component_movable_cells = 0
            # Use a set to store unique BLOCKED cells adjacent to this component
            current_component_adjacent_blocked = set()

            while q:
                curr_r, curr_c = q.popleft()
                current_component_movable_cells += 1 # This movable cell is part of the component

                # Explore neighbors
                for i in range(4):
                    nr, nc = curr_r + dr[i], curr_c + dc[i]
                    
                    # Check if neighbor is within grid boundaries
                    if 0 <= nr < H and 0 <= nc < W:
                        if cell_type[nr][nc] == 'MOVABLE' and not visited[nr][nc]:
                            # If neighbor is MOVABLE and not yet visited, add to queue for further exploration
                            visited[nr][nc] = True
                            q.append((nr, nc))
                        elif cell_type[nr][nc] == 'BLOCKED':
                            # If neighbor is BLOCKED, it's reachable but no further movement from it.
                            # Add its coordinates as a tuple to the set of adjacent blocked cells for this component.
                            current_component_adjacent_blocked.add((nr, nc))
            
            # Calculate total reachable cells for this component
            # Sum of MOVABLE cells in the component + unique BLOCKED cells adjacent to the component
            total_reachable_cells = current_component_movable_cells + len(current_component_adjacent_blocked)
            
            # Update the overall maximum degree of freedom
            max_degree_of_freedom = max(max_degree_of_freedom, total_reachable_cells)

# Print the final result
print(max_degree_of_freedom)