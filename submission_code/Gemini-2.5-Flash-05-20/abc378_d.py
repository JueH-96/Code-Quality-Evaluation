import sys

# Set a higher recursion limit. K=11 means a path of length 12, so recursion
# depth can go up to 11 (for moves_made from 0 to 11). Python's default limit
# (often 1000) is usually sufficient, but increasing it is a good safeguard.
sys.setrecursionlimit(2000) 

# Read H, W, K from the first line
H, W, K = map(int, sys.stdin.readline().split())

# Read the grid
grid = []
for _ in range(H):
    grid.append(sys.stdin.readline().strip())

# Memoization table to store results of dfs calls
# Key: (r, c, moves_made, visited_mask)
# Value: Number of valid paths from this state
memo = {}

# Directions for movement: (dr, dc) for up, down, left, right
DR = [-1, 1, 0, 0]  # Row changes
DC = [0, 0, -1, 1]  # Column changes

def dfs(r, c, moves_made, visited_mask):
    """
    Recursively counts the number of ways to complete a path of K moves,
    starting from an initial cell, currently at (r, c), having made 'moves_made'
    moves, with cells specified in 'visited_mask' already visited.
    """
    
    # Create a unique state key for memoization
    state = (r, c, moves_made, visited_mask)
    
    # If this state has been computed before, return the stored result
    if state in memo:
        return memo[state]

    # Base case: If K moves have been successfully made, this is one valid path
    if moves_made == K:
        return 1

    current_paths = 0
    
    # Explore all four adjacent cells (up, down, left, right)
    for i in range(4):
        nr, nc = r + DR[i], c + DC[i]

        # Check boundary conditions: ensure the next cell is within the grid
        if not (0 <= nr < H and 0 <= nc < W):
            continue
        
        # Check if the next cell is blocked by '#'
        if grid[nr][nc] == '#':
            continue
        
        # Calculate the unique index for the next cell (nr, nc)
        # This index will be used for the bitmask
        next_cell_idx = nr * W + nc
        
        # Check if the next cell has already been visited in the current path.
        # This is done by checking if the corresponding bit in visited_mask is set.
        if (visited_mask >> next_cell_idx) & 1:  # (visited_mask & (1 << next_cell_idx)) != 0
            continue
        
        # If the next cell is valid and unvisited, mark it as visited for the recursive call
        new_visited_mask = visited_mask | (1 << next_cell_idx)
        
        # Recursively call dfs for the next cell and add its results
        current_paths += dfs(nr, nc, moves_made + 1, new_visited_mask)
    
    # Store the computed result in the memoization table before returning
    memo[state] = current_paths
    return current_paths

total_valid_paths = 0

# Iterate over all possible starting cells in the grid
for r0 in range(H):
    for c0 in range(W):
        # A path can only start from an empty cell ('.')
        if grid[r0][c0] == '.':
            # Calculate the initial visited mask for the starting cell.
            # The bit corresponding to (r0, c0) is set.
            initial_visited_mask = (1 << (r0 * W + c0))
            
            # Start the DFS from this initial cell.
            # moves_made is 0 because no moves have been made yet (it's the first cell).
            total_valid_paths += dfs(r0, c0, 0, initial_visited_mask)

# Print the final count of all valid paths
print(total_valid_paths)