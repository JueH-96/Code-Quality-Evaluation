# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    """
    Reads the grid dimensions and state, finds the location of the missing cookie,
    and prints its 1-based coordinates.
    """
    # Read dimensions H (height) and W (width) of the grid from standard input
    H, W = map(int, sys.stdin.readline().split())

    # Read the grid state into a list of strings. Each string represents a row.
    # '#' denotes a cookie, '.' denotes an empty square.
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Initialize variables to store the boundaries (0-based indices) of the
    # rectangle containing '#' characters (cookies).
    # Initialize min values to be larger than any possible index,
    # and max values to be smaller than any possible index.
    min_r = H  # Minimum row index containing '#'
    max_r = -1 # Maximum row index containing '#'
    min_c = W  # Minimum column index containing '#'
    max_c = -1 # Maximum column index containing '#'

    # Iterate through each cell of the grid to find the bounding box of '#' characters.
    # The problem statement guarantees that the original rectangle of cookies had
    # height and width of at least 2. This property ensures that even after
    # one cookie is removed (potentially from a boundary), the minimum and maximum
    # row/column indices of the remaining '#' characters still accurately define
    # the boundaries of the original rectangle [a, b] x [c, d].
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '#':
                # Update the boundaries if a '#' is found
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    # Now, the variables `min_r`, `max_r`, `min_c`, `max_c` hold the 0-based
    # indices defining the rows (from `min_r` to `max_r`) and columns
    # (from `min_c` to `max_c`) of the original cookie rectangle.

    # Iterate through the cells within this identified bounding box.
    # According to the problem, exactly one cell within the original rectangle
    # will be empty ('.') because Snuke ate one cookie.
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            # Check if the current cell within the bounding box is empty.
            if grid[r][c] == '.':
                # This is the location of the eaten cookie.
                # The problem requires the output coordinates to be 1-based.
                # Convert the 0-based indices (r, c) to 1-based (r+1, c+1).
                print(f"{r + 1} {c + 1}")

                # Since the location of the eaten cookie is unique,
                # we can stop the search and return from the function.
                return

# Execute the solve function to run the program
solve()