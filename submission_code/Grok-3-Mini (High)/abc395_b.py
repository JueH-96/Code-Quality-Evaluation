# Read N from input
N = int(input())

# Initialize a N x N grid with spaces
grid = [[' ' for _ in range(N)] for _ in range(N)]

# Perform operations for each i from 1 to N
for i in range(1, N + 1):
    j = N + 1 - i
    if i <= j:
        # Determine color based on i parity
        color = '#' if i % 2 == 1 else '.'
        # Calculate 0-based indices
        row_start = i - 1
        row_end = j - 1
        col_start = i - 1
        col_end = j - 1
        # Fill the rectangular region
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                grid[row][col] = color

# Output the grid
for row in grid:
    print(''.join(row))