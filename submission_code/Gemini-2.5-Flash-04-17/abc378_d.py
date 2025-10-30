import sys

# Increase recursion depth if needed. K=11 means path length 12, max recursion depth 12.
# Default might be ok, but setting higher is safer for recursive solutions.
# The maximum path length is K+1 = 12, meaning up to 12 recursive calls deep.
# Setting a limit slightly above 12 (e.g., 2000) is sufficient and safe.
sys.setrecursionlimit(2000)

H, W, K = map(int, sys.stdin.readline().split())
grid = [sys.stdin.readline().strip() for _ in range(H)]

# Map (r, c) to index r * W + c for bitmask
def get_index(r, c):
    return r * W + c

# Memoization dictionary
# Key: (current_row, current_col, steps_taken, visited_mask)
# Value: number of ways to complete the path from this state
memo = {}

# Directions: up, down, left, right (dr, dc)
dr = [-1, 1, 0, 0]
dc = [0, 0, -1, 1]

def solve(r, c, steps_taken, visited_mask):
    """
    Counts the number of simple paths of length K+1
    that start from an empty cell, reach cell (r, c) in steps_taken moves,
    and visit the cells specified by visited_mask.

    Args:
        r (int): Current row (0-indexed).
        c (int): Current column (0-indexed).
        steps_taken (int): Number of moves made so far.
        visited_mask (int): Bitmask representing visited cells.
                            The bit at index `get_index(row, col)` is 1
                            if cell (row, col) has been visited.

    Returns:
        int: The number of ways to complete the path to length K+1.
    """
    # State for memoization
    state = (r, c, steps_taken, visited_mask)

    # Base case: If we have made K steps, we have a path of length K+1.
    if steps_taken == K:
        return 1

    # Check memoization table
    if state in memo:
        return memo[state]

    count = 0

    # Explore possible next moves (neighbors)
    for i in range(4):
        nr = r + dr[i]
        nc = c + dc[i]

        # Check if the neighbor is within grid bounds
        if not (0 <= nr < H and 0 <= nc < W):
            continue

        # Check if the neighbor cell is blocked ('#')
        if grid[nr][nc] == '#':
            continue

        # Get the index of the neighbor cell
        neighbor_idx = get_index(nr, nc)

        # Check if the neighbor cell has already been visited in the current path
        # Use the bitmask: if the neighbor_idx-th bit is 1, it's visited
        if (visited_mask >> neighbor_idx) & 1:
            continue

        # If the neighbor is valid (in bounds, not blocked, not visited),
        # make a recursive call from the neighbor cell, having taken one more step.
        # Update the visited mask to include the neighbor cell.
        new_mask = visited_mask | (1 << neighbor_idx)
        count += solve(nr, nc, steps_taken + 1, new_mask)

    # Store the computed result in the memoization table before returning
    memo[state] = count
    return count

# Total number of ways to form a simple path of length K+1
total_paths = 0

# Iterate through all possible starting cells (i_0, j_0)
for r_start in range(H):
    for c_start in range(W):
        # A path must start from an empty cell '.'
        if grid[r_start][c_start] == '.':
            # Get the index of the starting cell
            start_idx = get_index(r_start, c_start)
            # The initial visited mask contains only the starting cell
            initial_mask = 1 << start_idx
            # Start the recursive process from the starting cell, with 0 steps taken
            # We need to make K moves in total (resulting in K+1 cells in the path)
            total_paths += solve(r_start, c_start, 0, initial_mask)

# Print the final count
print(total_paths)