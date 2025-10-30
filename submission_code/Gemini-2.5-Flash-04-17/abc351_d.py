import sys
from collections import deque

# Function to check if a cell is within grid bounds
def is_valid(r, c, H, W):
    return 0 <= r < H and 0 <= c < W

# Function to check if a cell (r, c) is movable from
# (i.e., it's empty and none of its neighbors are magnets)
def check_can_move_from(r, c, H, W, grid):
    # We only call this for '.' cells in the main loop, but add a check just in case
    if grid[r][c] == '#':
        return False

    # Check neighbors for magnets
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    for i in range(4):
        nr, nc = r + dr[i], c + dc[i]
        if is_valid(nr, nc, H, W) and grid[nr][nc] == '#':
            return False # Found an adjacent magnet

    return True # No adjacent magnets


# Read input
H, W = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(H)]

# Compute can_move_from status for all empty cells
# can_move[r][c] is True if cell (r, c) is '.' AND has no adjacent '#', False otherwise (if it's '.' with adjacent '#' or if it's '#')
can_move = [[False for _ in range(W)] for _ in range(H)]
for r in range(H):
    for c in range(W):
        if grid[r][c] == '.':
             can_move[r][c] = check_can_move_from(r, c, H, W, grid)

# visited_S tracks cells visited during BFS for S components (cells where can_move is True)
visited_S = [[False for _ in range(W)] for _ in range(H)]

max_degree = 0
dr = [-1, 1, 0, 0]
dc = [0, 0, -1, 1]

# Iterate through all cells
for r in range(H):
    for c in range(W):
        # Only consider empty cells as starting points
        if grid[r][c] == '.':
            # Case 1: Cannot move from this cell (it's empty but has adjacent magnets)
            # These cells belong to set U. Their DOF is 1 (the cell itself).
            if not can_move[r][c]:
                max_degree = max(max_degree, 1)
            # Case 2: Can move from this cell (it's empty and has no adjacent magnets)
            # These cells belong to set S. Process if not visited as part of an S component yet.
            elif not visited_S[r][c]:
                # Start BFS to find the S component connected to (r, c)
                q = deque([(r, c)])
                visited_S[r][c] = True
                current_S_component = set()
                current_S_component.add((r, c))

                while q:
                    curr_r, curr_c = q.popleft()

                    # Explore neighbors to find other cells within the same S component
                    # Movement is only allowed *from* cells where can_move is True.
                    # Reachable cells *within* S are those connected via paths where every cell *on the path* is in S.
                    for i in range(4):
                        nr, nc = curr_r + dr[i], curr_c + dc[i]
                        # A neighbor (nr, nc) is part of the same S component if it's valid,
                        # also allows movement (can_move[nr][nc] is True), and hasn't been visited in this BFS.
                        if is_valid(nr, nc, H, W) and can_move[nr][nc] and not visited_S[nr][nc]:
                            visited_S[nr][nc] = True
                            q.append((nr, nc))
                            current_S_component.add((nr, nc))

                # After finding the entire S component, find all adjacent empty cells in U
                # These are cells that can be reached in one step from a cell in the S component,
                # but from which no further moves are possible. They are part of the reachable set.
                reachable_U_cells = set()
                for sr, sc in current_S_component:
                    # Explore neighbors of cells within the S component
                    for i in range(4):
                        nr, nc = sr + dr[i], sc + dc[i]
                        # Check if neighbor is valid, is empty, and cannot move from (is in U)
                        # grid[nr][nc] == '.' ensures it's empty.
                        # not can_move[nr][nc] ensures it's in U (empty but can't move from).
                        # No need to check if it's already in the set, 'set' handles uniqueness.
                        if is_valid(nr, nc, H, W) and grid[nr][nc] == '.' and not can_move[nr][nc]:
                            reachable_U_cells.add((nr, nc))

                # The total degree of freedom for any starting cell within this S component
                # is the number of cells in the component itself plus all adjacent reachable U cells.
                current_degree = len(current_S_component) + len(reachable_U_cells)
                max_degree = max(max_degree, current_degree)

# Print the final result
print(max_degree)