# YOUR CODE HERE
import sys

def solve():
    # Read dimensions H (number of rows) and W (number of columns)
    H, W = map(int, sys.stdin.readline().split())
    
    # Read the grid state into a list of strings
    # Each string represents a row
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Initialize bounding box coordinates.
    # We need to find the minimum and maximum row and column indices
    # among all cells initially painted black ('#').
    # Initialize min coordinates to values larger than any possible index,
    # and max coordinates to values smaller than any possible index.
    # Python uses 0-based indexing, so valid row indices are 0..H-1, column indices 0..W-1.
    min_r = H  # Minimum row index of a '#' cell
    max_r = -1 # Maximum row index of a '#' cell
    min_c = W  # Minimum column index of a '#' cell
    max_c = -1 # Maximum column index of a '#' cell

    # Flag to track if at least one '#' cell is found.
    # The problem statement guarantees at least one '#' exists.
    found_hash = False 

    # Iterate through the entire grid to find all '#' cells
    # and update the bounding box coordinates accordingly.
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '#':
                found_hash = True
                # Update the minimum and maximum row indices
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                # Update the minimum and maximum column indices
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    # Since the problem guarantees at least one '#' cell exists,
    # `found_hash` will be True, and `min_r`, `max_r`, `min_c`, `max_c`
    # will hold the coordinates defining the minimal bounding box containing all '#' cells.

    # Assume initially that it is possible to form the required rectangle.
    possible = True
    
    # Check all cells within the determined minimal bounding box.
    # The bounding box spans rows from `min_r` to `max_r` inclusive,
    # and columns from `min_c` to `max_c` inclusive.
    # The Python `range(start, stop)` function generates numbers from `start` up to `stop-1`.
    # So, to include `max_r` and `max_c`, we use `max_r + 1` and `max_c + 1` as the `stop` values.
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            # If any cell inside this bounding box is initially painted white ('.'),
            # then it's impossible to form a solid black rectangle.
            # This is because any rectangle containing all '#' cells must contain this bounding box,
            # and thus must contain this '.' cell. But the '.' cell must be white.
            if grid[r][c] == '.':
                possible = False
                # If impossibility is detected, no need to check further cells inside the box.
                break # Exit the inner loop (over columns)
        if not possible:
            # If impossibility is detected in the inner loop, exit the outer loop as well.
            break # Exit the outer loop (over rows)

    # After checking all cells within the minimal bounding box:
    if possible:
        # If no '.' cell was found within the minimal bounding box, it means all cells
        # inside are either '#' or '?'. We can choose this minimal bounding box
        # as the target rectangle. Paint all '?' inside it black, and all '?' outside it white.
        # This construction results in a valid configuration where all black cells form a rectangle,
        # respecting the initial '#' and '.' cells.
        print("Yes")
    else:
        # If a '.' cell was found inside the minimal bounding box, it's impossible
        # to satisfy the condition that all cells within the final rectangle are black.
        print("No")

# Call the solve function to execute the logic
solve()