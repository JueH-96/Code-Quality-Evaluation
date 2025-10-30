# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem of finding the number of safe empty squares on an 8x8 grid.

    Reads the grid configuration from standard input, determines which rows and columns
    contain existing pieces, and counts the number of empty squares that are not
    in the same row or column as any existing piece. Prints the count to standard output.
    """
    # Read the 8x8 grid from standard input
    grid = [sys.stdin.readline().strip() for _ in range(8)]

    # Initialize boolean arrays to track rows and columns containing pieces
    # row_has_piece[r] will be True if row r (0-indexed) contains a piece '#'
    # col_has_piece[c] will be True if column c (0-indexed) contains a piece '#'
    row_has_piece = [False] * 8
    col_has_piece = [False] * 8

    # First pass: Iterate through the grid to identify rows and columns with pieces
    for r in range(8):
        for c in range(8):
            if grid[r][c] == '#':
                # Mark the row and column as having a piece
                row_has_piece[r] = True
                col_has_piece[c] = True

    # Initialize the count of safe squares
    count = 0

    # Second pass: Iterate through the grid again to find safe empty squares
    for r in range(8):
        for c in range(8):
            # Check if the current square is empty
            if grid[r][c] == '.':
                # Check if the row and column are safe (do not contain any pieces)
                # A square (r, c) is safe if row r has no pieces AND column c has no pieces.
                if not row_has_piece[r] and not col_has_piece[c]:
                    # If the square is empty and safe, increment the count
                    count += 1

    # Print the final count of safe empty squares
    print(count)

# Call the solve function to execute the logic
solve()