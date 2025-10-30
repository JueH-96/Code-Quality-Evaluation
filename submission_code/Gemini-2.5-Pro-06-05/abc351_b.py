# YOUR CODE HERE
import sys

def solve():
    """
    Reads two NxN grids and finds the single cell where they differ.
    """
    # Read the size of the grid from standard input.
    try:
        n = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # This handles cases where input is not a valid integer or is empty.
        # It's defensive programming, unlikely to be triggered by valid test cases.
        return

    # Read grid A into a list of strings. Each string represents a row.
    grid_a = [sys.stdin.readline().strip() for _ in range(n)]

    # Read grid B similarly.
    grid_b = [sys.stdin.readline().strip() for _ in range(n)]

    # Iterate through each cell of the grids to find the difference.
    # The loops use 0-based indexing (i, j from 0 to n-1).
    for i in range(n):
        for j in range(n):
            # Compare the characters at the current (i, j) position.
            if grid_a[i][j] != grid_b[i][j]:
                # A difference is found. The problem asks for 1-based coordinates.
                # So, we add 1 to our 0-based indices.
                print(f"{i + 1} {j + 1}")
                
                # As per the problem statement, there's exactly one differing cell.
                # Once found, we can terminate the program.
                return

# Run the solution
solve()