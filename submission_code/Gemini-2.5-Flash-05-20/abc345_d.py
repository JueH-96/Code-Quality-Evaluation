import sys

# Increase the recursion limit for potentially deep recursive calls.
# H*W can be up to 100, meaning recursion depth could be 100+ for 1x1 tiles.
sys.setrecursionlimit(2000)

def solve(grid, tiles_data, used_mask, H, W, current_r, current_c):
    """
    Attempts to tile the grid recursively, starting the search for the next
    uncovered cell from (current_r, current_c).

    Args:
        grid (list[list[bool]]): The current state of the grid. True if cell is covered, False otherwise.
        tiles_data (list[tuple[int, int]]): A list of (height, width) dimensions for each tile.
        used_mask (int): A bitmask indicating which tiles have been used in the current path.
        H (int): Grid height.
        W (int): Grid width.
        current_r (int): The starting row for finding the next uncovered cell.
        current_c (int): The starting column for finding the next uncovered cell.

    Returns:
        bool: True if a valid tiling is found from this state, False otherwise.
    """

    # Find the first uncovered cell (r_to_place, c_to_place)
    # searching row by row, column by column, starting from (current_r, current_c).
    r_to_place, c_to_place = -1, -1
    r_search, c_search = current_r, current_c

    while r_search < H:
        if c_search < W:
            if not grid[r_search][c_search]:
                r_to_place, c_to_place = r_search, c_search
                break # Found the first uncovered cell
            c_search += 1
        else: # Move to the next row if columns are exhausted
            c_search = 0
            r_search += 1
    
    # Base case: If no uncovered cell is found, the entire grid is covered.
    if r_to_place == -1:
        return True

    # Iterate through all available tiles (not yet used in the current path)
    for tile_idx in range(len(tiles_data)):
        # Check if this tile has already been used (bit is set in used_mask)
        if (used_mask >> tile_idx) & 1:
            continue # Skip already used tiles

        tile_dims = tiles_data[tile_idx]

        # Consider both original and rotated orientations for the tile
        orientations = [(tile_dims[0], tile_dims[1])]
        if tile_dims[0] != tile_dims[1]: # Only add rotated if it's not a square
            orientations.append((tile_dims[1], tile_dims[0]))

        for tile_h, tile_w in orientations:
            # Check if the tile fits within the grid boundaries when placed at (r_to_place, c_to_place)
            if r_to_place + tile_h <= H and c_to_place + tile_w <= W:
                # Check for overlaps with already covered cells in the grid
                can_place = True
                for i_offset in range(tile_h):
                    for j_offset in range(tile_w):
                        if grid[r_to_place + i_offset][c_to_place + j_offset]:
                            can_place = False # Overlap detected
                            break
                    if not can_place:
                        break # No need to check further offsets

                if can_place:
                    # Place the tile: Mark all cells covered by this tile as True
                    for i_offset in range(tile_h):
                        for j_offset in range(tile_w):
                            grid[r_to_place + i_offset][c_to_place + j_offset] = True

                    # Recurse: Try to fill the rest of the grid.
                    # Pass (r_to_place, c_to_place) to the next call to efficiently
                    # continue the search for the next empty cell from this point.
                    new_used_mask = used_mask | (1 << tile_idx) # Mark current tile as used
                    if solve(grid, tiles_data, new_used_mask, H, W, r_to_place, c_to_place):
                        return True # Solution found in a deeper call!

                    # Backtrack: Unplace the tile by marking cells as False.
                    # This is crucial to explore other possibilities in the search tree.
                    for i_offset in range(tile_h):
                        for j_offset in range(tile_w):
                            grid[r_to_place + i_offset][c_to_place + j_offset] = False
    
    # If no tile could be placed at (r_to_place, c_to_place) to form a complete solution,
    # or if all possible placements from this state have been tried and failed.
    return False

# Read input from standard input
N, H, W = map(int, sys.stdin.readline().split())

tiles_data = []
for _ in range(N):
    A, B = map(int, sys.stdin.readline().split())
    tiles_data.append((A, B))

# Initialize the grid with all cells as False (uncovered)
initial_grid = [[False for _ in range(W)] for _ in range(H)]

# Start the backtracking search from the top-left corner (0,0) with no tiles used (mask 0)
if solve(initial_grid, tiles_data, 0, H, W, 0, 0):
    print("Yes")
else:
    print("No")