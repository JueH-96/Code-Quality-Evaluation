# YOUR CODE HERE
def count_safe_squares(grid):
    rows, cols = 8, 8
    unsafe_rows = set()
    unsafe_cols = set()

    # Mark unsafe rows and columns
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '#':
                unsafe_rows.add(i)
                unsafe_cols.add(j)

    # Count safe squares
    safe_count = 0
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '.' and i not in unsafe_rows and j not in unsafe_cols:
                safe_count += 1

    return safe_count

# Read input
grid = [input().strip() for _ in range(8)]

# Calculate and print the result
result = count_safe_squares(grid)
print(result)