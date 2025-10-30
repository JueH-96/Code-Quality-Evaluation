import sys

# Set higher recursion depth if needed, usually not required for iterative solutions like this.
# sys.setrecursionlimit(4000010) 

def solve():
    """
    Solves the cookie removal problem based on the given specification.
    Reads grid dimensions H, W and the grid itself from stdin.
    Simulates the cookie removal process iteratively.
    Prints the final count of remaining cookies to stdout.
    """
    H, W = map(int, sys.stdin.readline().split())
    # Read grid into a list of lists of characters. '.' will represent removed cookies.
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    # The main loop simulates the iterative removal process.
    # It continues as long as cookies are marked and removed in an iteration.
    while True:
        # Use a set to store coordinates (r, c) of cookies marked for removal in this iteration.
        # Sets handle uniqueness automatically and provide efficient additions.
        marked_coords = set() 

        # Step 1: Check rows for marking conditions.
        # Iterate through each row index r from 0 to H-1.
        for r in range(H):
            first_color = None # Store the color of the first cookie encountered in the row.
            count = 0 # Count the number of cookies remaining in the row.
            coords_in_row = [] # Store coordinates (r, c) of cookies present in this row.
            
            # First pass over the row: Collect info about existing cookies.
            for c in range(W):
                if grid[r][c] != '.': # Check if cookie exists (not already removed).
                    if count == 0:
                        first_color = grid[r][c] # Record the color of the first cookie found.
                    count += 1
                    coords_in_row.append((r, c)) # Store coordinates for potential marking later.
            
            # Check if the row satisfies the marking conditions:
            # Condition 1: There are two or more cookies remaining in the row (count >= 2).
            # Condition 2: All remaining cookies in the row have the same color.
            if count >= 2:
                all_same = True
                # Second pass (verification): Check if all cookies found indeed have the 'first_color'.
                # This check iterates over the coordinates stored earlier.
                for current_r, current_c in coords_in_row: 
                    # Access the grid directly to ensure we check the current state.
                    if grid[current_r][current_c] != first_color:
                        all_same = False # Found a cookie with a different color.
                        break
                
                if all_same:
                    # If conditions are met, mark all cookies currently in this row.
                    # Add their coordinates to the set 'marked_coords'.
                    for R_coord, C_coord in coords_in_row:
                         marked_coords.add((R_coord, C_coord))

        # Step 2: Check columns for marking conditions.
        # Iterate through each column index c from 0 to W-1.
        for c in range(W):
            first_color = None # Store the color of the first cookie encountered in the column.
            count = 0 # Count the number of cookies remaining in the column.
            coords_in_col = [] # Store coordinates (r, c) of cookies present in this column.

            # First pass over the column: Collect info about existing cookies.
            for r in range(H):
                 if grid[r][c] != '.': # Check if cookie exists.
                     if count == 0:
                         first_color = grid[r][c] # Record first color found.
                     count += 1
                     coords_in_col.append((r, c)) # Store coordinates.
            
            # Check if the column satisfies the marking conditions.
            if count >= 2:
                 all_same = True
                 # Second pass (verification): Check uniformity.
                 for current_r, current_c in coords_in_col: 
                     # Access grid directly for current state check.
                     if grid[current_r][current_c] != first_color:
                         all_same = False # Found different color.
                         break
                
                 if all_same:
                     # If conditions met, mark all cookies currently in this column.
                     for R_coord, C_coord in coords_in_col:
                         marked_coords.add((R_coord, C_coord))

        # Step 3: Check for termination condition and remove marked cookies.
        if not marked_coords:
            # If the set 'marked_coords' is empty, it means no cookies were marked in this iteration.
            # According to the procedure description, we terminate the process.
            break 

        # If there were marked cookies (marked_coords is not empty), remove them from the grid.
        # Iterate through the coordinates stored in the set.
        for r, c in marked_coords:
             # Double-check if the cookie at (r, c) hasn't already been removed.
             # This check adds robustness, although theoretically, coordinates in marked_coords
             # should correspond to existing cookies at the start of the iteration.
             if grid[r][c] != '.': 
                 grid[r][c] = '.' # Mark the cookie position as empty (removed).

    # After the loop terminates (no more cookies marked), count the remaining cookies.
    final_count = 0
    for r in range(H):
        for c in range(W):
            if grid[r][c] != '.': # If the position is not marked as empty...
                final_count += 1 # ...increment the count of remaining cookies.

    # Print the final count to standard output.
    print(final_count)

# Call the main function to execute the solution logic.
solve()