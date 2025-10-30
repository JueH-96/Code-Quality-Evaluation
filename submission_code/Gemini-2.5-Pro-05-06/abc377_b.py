def solve():
    grid = []
    for _ in range(8):
        # Read each row of the grid. input() reads a line from stdin.
        # Standard behavior is that it includes the newline if not stripped,
        # but for fixed-length strings and checks like `in` or indexing within bounds,
        # it's often fine. Assuming standard contest environment behavior where
        # input() provides the string content correctly.
        grid.append(input())

    # Determine safe rows
    safe_rows = [True] * 8  # Initialize all rows as safe
    for r in range(8):
        # If a '#' is found in the string representing the row, it's not safe
        if '#' in grid[r]:
            safe_rows[r] = False

    # Determine safe columns
    safe_cols = [True] * 8  # Initialize all columns as safe
    for c in range(8):  # Iterate through column indices
        for r in range(8):  # Iterate through row indices for the current column
            if grid[r][c] == '#':
                # If a '#' is found in this column, it's not safe
                safe_cols[c] = False
                break  # No need to check further for this column

    # Count the number of safe rows
    # In Python, sum() on a list of booleans treats True as 1 and False as 0
    num_safe_r = sum(safe_rows)
    
    # Count the number of safe columns
    num_safe_c = sum(safe_cols)
    
    # The total number of squares (R,C) where row R is safe and column C is safe
    # is the product of the number of safe rows and safe columns.
    # As discussed, if row R and column C are safe, grid[R][C] must be '.',
    # so the "place on an empty square" condition is met.
    count = num_safe_r * num_safe_c
    
    print(count)

if __name__ == '__main__':
    solve()