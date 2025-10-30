# YOUR CODE HERE
import sys

def main():
    """
    Reads the grid state and identifies the location of the eaten cookie.
    """
    # Read grid dimensions from standard input.
    try:
        H, W = map(int, sys.stdin.readline().split())
    except (ValueError, IndexError):
        # Handle case of empty or malformed input line.
        return

    # Read the grid into a list of strings.
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Calculate the number of cookies ('#') in each row and column.
    # For rows, we can use the efficient string.count() method.
    row_counts = [row.count('#') for row in grid]
    
    # For columns, a simple iteration is needed.
    col_counts = [0] * W
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '#':
                col_counts[c] += 1

    # Determine the original dimensions of the cookie rectangle.
    # Since the rectangle's height and width are at least 2, at least one row
    # and one column will remain "full" after one cookie is eaten.
    # The max count in any row gives the original width, and the max in any
    # column gives the original height.
    rect_width = max(row_counts)
    rect_height = max(col_counts)

    # Find the row and column where the cookie was eaten.
    # The affected row will have `rect_width - 1` cookies.
    # The affected column will have `rect_height - 1` cookies.
    # The answer is guaranteed to be unique.
    
    eaten_row = -1
    for r, count in enumerate(row_counts):
        # Find the row that is part of the rectangle but is not full.
        # The count must also be non-zero to distinguish from empty rows.
        if 0 < count < rect_width:
            eaten_row = r
            break
            
    eaten_col = -1
    for c, count in enumerate(col_counts):
        # Find the column that is part of the rectangle but is not full.
        if 0 < count < rect_height:
            eaten_col = c
            break

    # Print the result using 1-based indexing as required.
    print(eaten_row + 1, eaten_col + 1)

if __name__ == "__main__":
    main()