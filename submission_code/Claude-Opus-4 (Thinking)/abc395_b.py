# YOUR CODE HERE
N = int(input())

# Initialize grid
grid = [['#' for _ in range(N)] for _ in range(N)]

# Perform operations
for i in range(1, N + 1):
    j = N + 1 - i
    if i <= j:
        # Fill rectangle from (i,i) to (j,j)
        color = '#' if i % 2 == 1 else '.'
        for row in range(i - 1, j):  # Convert to 0-indexed
            for col in range(i - 1, j):  # Convert to 0-indexed
                grid[row][col] = color

# Print the grid
for row in grid:
    print(''.join(row))