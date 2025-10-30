# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input grids A and B, finds the single cell where they differ,
    and prints the 1-based row and column indices of that cell.
    """
    # Read the size of the grid
    # N represents the number of rows and columns
    n = int(sys.stdin.readline())

    # Read grid A
    # grid_a is stored as a list of strings, where each string represents a row.
    grid_a = [sys.stdin.readline().strip() for _ in range(n)]

    # Read grid B
    # grid_b is stored similarly to grid_a.
    grid_b = [sys.stdin.readline().strip() for _ in range(n)]

    # Iterate through each cell of the grids to find the difference
    # The outer loop iterates through rows (0-based index r)
    for r in range(n):
        # The inner loop iterates through columns (0-based index c)
        for c in range(n):
            # Compare the characters at the current cell (r, c) in both grids
            # grid_a[r] accesses the r-th row (string) of grid A
            # grid_a[r][c] accesses the character at column c in row r of grid A
            if grid_a[r][c] != grid_b[r][c]:
                # If the characters are different, we have found the unique differing cell.
                # The problem guarantees exactly one such cell exists.

                # Print the 1-based row index (r + 1) and 1-based column index (c + 1)
                # The output format requires indices separated by a space.
                print(f"{r + 1} {c + 1}")

                # Since the unique difference is found, we can stop the search and exit the function.
                return

# Call the solve function to execute the main logic of the program.
# This function will handle reading input, processing, and printing the output.
solve()