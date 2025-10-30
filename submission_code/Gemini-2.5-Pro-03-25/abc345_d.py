# YOUR CODE HERE
import sys

# Set a higher recursion depth limit. This is sometimes necessary for deep recursive calls
# in competitive programming, although the maximum depth here seems limited by N=7 tiles.
# Setting it to a large value like 3000 provides a safety margin.
try:
    sys.setrecursionlimit(3000) 
except Exception as e: 
    # Some environments might restrict changing the recursion limit.
    # If an error occurs (e.g., ReadOnlyFilesystemError on certain platforms), 
    # we just continue with the default limit.
    pass 

# Global variables to store input and memoization table
N = 0  # Number of tiles
H = 0  # Grid height
W = 0  # Grid width
tiles = []  # List to store tile dimensions (Ai, Bi)
memo = {}  # Dictionary for memoization: stores results of subproblems

# Grid state representation:
# The grid is represented as a flat tuple of integers (0 or 1).
# A cell at row `r`, column `c` corresponds to index `r * W + c` in the tuple.
# 0 means the cell is uncovered, 1 means it's covered.
# Using a tuple makes the state hashable, which is required for dictionary keys (memoization).

def solve(grid_tuple, used_mask):
    """
    Recursive backtracking function to determine if the grid can be tiled using a subset of the given tiles.
    Uses memoization to cache results of previously computed states.
    
    Args:
        grid_tuple (tuple): A tuple representing the current state of the grid (flattened).
                            0 indicates an uncovered cell, 1 indicates a covered cell.
        used_mask (int): An integer used as a bitmask. The i-th bit is set to 1 if the i-th tile 
                         (from the input list `tiles`) has been used, 0 otherwise.
        
    Returns:
        bool: True if the grid can be completely tiled starting from the current `grid_tuple` state 
              using a subset of the remaining available tiles. False otherwise.
    """
    # Create a state identifier tuple using the grid state and the mask of used tiles.
    state = (grid_tuple, used_mask)
    
    # Check if this state has already been computed and stored in the memoization table.
    if state in memo:
        # If yes, return the cached result directly.
        return memo[state]

    # Find the index of the first uncovered cell (value 0) in the grid tuple.
    # This corresponds to the top-leftmost uncovered cell using row-major order.
    first_uncovered_idx = -1
    try:
        first_uncovered_idx = grid_tuple.index(0)
    except ValueError:
        # If `grid_tuple.index(0)` raises ValueError, it means there are no 0s in the tuple.
        # This indicates that all cells are covered.
        pass

    # Base case for recursion: If no uncovered cell is found, the grid is fully tiled.
    if first_uncovered_idx == -1:
        # Successfully covered the entire grid. Store True for this state and return True.
        memo[state] = True
        return True

    # Calculate the row `r` and column `c` coordinates of the first uncovered cell.
    r = first_uncovered_idx // W
    c = first_uncovered_idx % W

    # Convert the grid tuple to a list to allow modification (placing/removing tiles).
    grid = list(grid_tuple)

    # Initialize the result for the current state as False. It will be set to True if any placement leads to a solution.
    res = False 
    
    # Iterate through all available tiles (indices 0 to N-1).
    for i in range(N):
        # Check if the i-th tile is unused by looking at the i-th bit in `used_mask`.
        if not (used_mask & (1 << i)):
            # Get the dimensions of the i-th tile.
            Ai, Bi = tiles[i]
            
            # --- Try placing the tile with orientation Ai x Bi ---
            
            # Check if the tile fits within the grid boundaries starting at (r, c).
            if r + Ai <= H and c + Bi <= W:
                can_place = True  # Flag to check if the area is clear.
                coords_to_fill = [] # List to store indices of cells to be covered by this tile.
                
                # Check if all cells within the target area (Ai x Bi) are currently uncovered (value 0).
                for row in range(r, r + Ai):
                    for col in range(c, c + Bi):
                        idx = row * W + col
                        if grid[idx] == 1: # If any cell is already covered (value 1)
                            can_place = False # Cannot place the tile here due to overlap.
                            break
                        coords_to_fill.append(idx) # Add index to list if cell is clear.
                    if not can_place: # If overlap detected, stop checking this area.
                        break
                
                # If the entire area is clear (can_place is True)
                if can_place:
                    # Place the tile: Mark all cells in `coords_to_fill` as covered (set value to 1).
                    for idx in coords_to_fill:
                        grid[idx] = 1
                    
                    # Recursively call `solve` with the updated grid state (as a tuple) and updated used mask.
                    # The new mask has the i-th bit set: `used_mask | (1 << i)`.
                    if solve(tuple(grid), used_mask | (1 << i)):
                        # If the recursive call returns True, it means a valid tiling was found down this path.
                        # Set the result for the current state `res` to True.
                        res = True
                    
                    # Backtrack: Regardless of the recursive call result, undo the placement.
                    # Reset the cells covered by this tile back to uncovered (value 0).
                    # This is necessary to explore other possibilities (e.g., using a different tile at (r,c)).
                    for idx in coords_to_fill:
                        grid[idx] = 0 
                    
                    # Optimization: If a solution was found (`res` is True), we don't need to try other tiles 
                    # or orientations from this state. We can break the loop and return True.
                    if res:
                        break # Break from the loop iterating through tiles (for i in range(N))

            # If a solution was found using the first orientation, break the loop early.
            if res:
                break
            
            # --- Try placing the tile with orientation Bi x Ai (if tile is not square) ---
            
            # Only attempt rotation if Ai is not equal to Bi.
            if Ai != Bi:
                 # Check if the rotated tile fits within the grid boundaries.
                 if r + Bi <= H and c + Ai <= W:
                    can_place = True # Reset flag for checking the rotated placement area.
                    coords_to_fill = [] # Reset list for coordinates.
                    
                    # Check if all cells within the target area (Bi x Ai) are currently uncovered.
                    for row in range(r, r + Bi):
                        for col in range(c, c + Ai):
                           idx = row * W + col
                           if grid[idx] == 1: # Check for overlap.
                               can_place = False
                               break
                           coords_to_fill.append(idx)
                        if not can_place:
                            break

                    # If the area for the rotated tile is clear.
                    if can_place:
                        # Place the rotated tile: Mark cells as covered.
                        for idx in coords_to_fill:
                            grid[idx] = 1
                        
                        # Recursively call `solve` with the updated state.
                        if solve(tuple(grid), used_mask | (1 << i)):
                            # If a solution is found.
                            res = True
                        
                        # Backtrack: Undo the placement of the rotated tile.
                        for idx in coords_to_fill:
                           grid[idx] = 0 
                    
                    # Optimization: If a solution was found, break the loop.
                    if res:
                        break # Break from the loop iterating through tiles (for i in range(N))

        # Optimization: If a solution was found with tile `i` (either orientation), stop checking subsequent tiles.
        if res:
            break

    # After trying all possibilities from the current state (all unused tiles at the first uncovered cell):
    # Store the final result (`res`, which is True if any path led to a solution, False otherwise) 
    # in the memoization table for this state.
    memo[state] = res
    # Return the result for this state.
    return res


def main():
    """
    Main function to handle input reading, initialization, calling the solver, and printing the output.
    """
    global N, H, W, tiles, memo # Declare usage of global variables
    
    # Read N, H, W from the first line of standard input.
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    H = int(line1[1])
    W = int(line1[2])
    
    # Read the dimensions (Ai, Bi) for each of the N tiles from subsequent lines.
    tiles = []
    for i in range(N):
        line = sys.stdin.readline().split()
        # Store tile dimensions as a tuple in the `tiles` list.
        tiles.append((int(line[0]), int(line[1])))

    # Initialize the grid state as a flat tuple of size H*W. All cells are initially uncovered (0).
    initial_grid_tuple = tuple([0] * (H * W))
    
    # Start the backtracking search process by calling the `solve` function.
    # The initial call uses the empty grid (`initial_grid_tuple`) and a mask of 0 (no tiles used yet).
    if solve(initial_grid_tuple, 0):
        # If `solve` returns True, it means a valid tiling is possible.
        print("Yes")
    else:
        # If `solve` returns False, it means no valid tiling could be found.
        print("No")

# Standard Python construct to ensure the `main` function is called only when the script is executed directly.
if __name__ == '__main__':
    main()