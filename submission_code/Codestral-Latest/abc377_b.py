# YOUR CODE HERE
import sys

def count_safe_squares(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Initialize sets to keep track of rows and columns with pieces
    rows_with_pieces = set()
    cols_with_pieces = set()

    # Populate the sets
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '#':
                rows_with_pieces.add(i)
                cols_with_pieces.add(j)

    # Count the safe squares
    safe_count = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '.' and i not in rows_with_pieces and j not in cols_with_pieces:
                safe_count += 1

    return safe_count

# Read input
grid = [input().strip() for _ in range(8)]

# Calculate and print the result
print(count_safe_squares(grid))