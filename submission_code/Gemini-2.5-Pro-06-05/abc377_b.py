# YOUR CODE HERE
import sys

def solve():
    """
    Reads the grid state from stdin, calculates the number of safe squares to place a piece,
    and prints the result to stdout.
    """
    
    # We will use boolean lists to keep track of which rows and columns
    # contain at least one piece ('#').
    # A value of `False` indicates a "safe" row/column (no pieces).
    # A value of `True` indicates an "attacked" row/column (at least one piece).
    rows_with_pieces = [False] * 8
    cols_with_pieces = [False] * 8

    # Iterate through each of the 8 rows of the grid.
    for r in range(8):
        # Read one row from standard input.
        line = sys.stdin.readline().strip()
        
        # Iterate through each of the 8 columns in the current row.
        for c in range(8):
            # Check if the square at (r, c) has a piece.
            if line[c] == '#':
                # If a piece is found, mark its row and column as having a piece.
                rows_with_pieces[r] = True
                cols_with_pieces[c] = True

    # A square (r, c) is a valid placement if row `r` has no pieces AND column `c` has no pieces.
    # The condition that the square must be empty is automatically met if its row has no pieces.

    # Count the number of rows that are "safe" (have no pieces).
    num_safe_rows = rows_with_pieces.count(False)

    # Count the number of columns that are "safe" (have no pieces).
    num_safe_cols = cols_with_pieces.count(False)

    # The total number of safe squares is the product of the number of safe rows
    # and the number of safe columns, as any combination of a safe row and a
    # safe column yields a safe square.
    result = num_safe_rows * num_safe_cols

    # Print the final result to standard output.
    print(result)

# Run the solution
solve()