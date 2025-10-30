# YOUR CODE HERE
import sys

# Read the 8x8 grid from standard input.
# Each line of input represents a row in the grid.
# The grid state is represented by 8 strings of length 8.
# S_i is the i-th string (1-indexed in problem), which corresponds to row i-1 (0-indexed).
grid = [sys.stdin.readline().strip() for _ in range(8)]

# Initialize boolean lists to keep track of whether a row or a column contains a piece ('#').
# row_has_piece[r] will be True if the row with index r (0-indexed) has at least one piece.
# col_has_piece[c] will be True if the column with index c (0-indexed) has at least one piece.
row_has_piece = [False] * 8
col_has_piece = [False] * 8

# Iterate through the grid to detect rows and columns containing pieces.
# If grid[r][c] == '#', it means there is a piece at row r and column c (0-indexed).
# This piece affects row r and column c for potential captures.
for r in range(8):
    for c in range(8):
        if grid[r][c] == '#':
            row_has_piece[r] = True  # Mark row r as having a piece
            col_has_piece[c] = True  # Mark column c as having a piece

# Initialize a counter for the number of squares where a new piece can be safely placed.
safe_squares_count = 0

# Iterate through all possible squares (r, c) in the 8x8 grid.
# r represents the row index (0 to 7), c represents the column index (0 to 7).
# This corresponds to square (r+1, c+1) in 1-indexed problem notation.
for r in range(8):
    for c in range(8):
        # A square (r, c) is a safe place to put a new piece if:
        # 1. The square (r, c) must be empty ('.').
        # 2. The new piece placed at (r, c) must not be captured by any existing piece.
        # A piece at (r, c) is captured if there is *any* existing piece in row r OR in column c.
        # Therefore, a piece at (r, c) is NOT captured if there are NO existing pieces in row r AND NO existing pieces in column c.

        # If row r has no pieces, row_has_piece[r] is False.
        # If column c has no pieces, col_has_piece[c] is False.

        # If row r has no pieces (row_has_piece[r] is False), then every square in row r must be empty ('.').
        # If column c has no pieces (col_has_piece[c] is False), then every square in column c must be empty ('.').
        # If both row_has_piece[r] and col_has_piece[c] are False, then square (r, c) must be empty ('.')
        # and there are no pieces in row r or column c to capture a new piece placed there.
        # Thus, checking that row_has_piece[r] is False AND col_has_piece[c] is False is sufficient
        # to identify squares that are both empty and safe from capture.
        if not row_has_piece[r] and not col_has_piece[c]:
            safe_squares_count += 1 # This square is a safe placement spot.

# Print the final count of safe squares.
print(safe_squares_count)