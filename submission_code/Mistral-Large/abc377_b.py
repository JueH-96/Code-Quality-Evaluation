import sys

def count_safe_squares(grid):
    rows = len(grid)
    cols = len(grid[0])

    # Check if a square is safe
    def is_safe(i, j):
        # Check row and column
        for k in range(cols):
            if grid[i][k] == '#':
                return False
        for k in range(rows):
            if grid[k][j] == '#':
                return False
        return True

    safe_count = 0

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '.' and is_safe(i, j):
                safe_count += 1

    return safe_count

# Read input
grid = [sys.stdin.readline().strip() for _ in range(8)]

# Calculate and print the result
print(count_safe_squares(grid))