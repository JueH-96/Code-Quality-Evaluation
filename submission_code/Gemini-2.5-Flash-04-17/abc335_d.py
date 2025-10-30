# Read N
N = int(input())

# Create grid and initialize with a placeholder (like 0)
grid = [[0] * N for _ in range(N)]

# The middle index (0-based)
mid = N // 2

# Fill the grid using a layer-by-layer spiral pattern
num = 1
# k represents the depth of the current spiral layer, starting from the outermost layer (k=0)
# The layers go inwards up to the layer just before the center cell.
# For an N x N grid (N odd), there are (N-1)/2 layers before the central cell.
# N // 2 gives (N-1)/2 for odd N.
for k in range(N // 2):
    # Fill the top edge of the current layer (moving right)
    # The row is fixed at k. Columns go from k to N-1-k.
    # In Python range, the end is exclusive, so N-k is used for N-1-k.
    r = k
    for c in range(k, N - k):
        grid[r][c] = num
        num += 1

    # Fill the right edge of the current layer (moving down)
    # The column is fixed at N-1-k. Rows go from k+1 to N-1-k.
    c = N - 1 - k
    for r in range(k + 1, N - k):
        grid[r][c] = num
        num += 1

    # Fill the bottom edge of the current layer (moving left)
    # The row is fixed at N-1-k. Columns go from N-1-k-1 down to k.
    # In Python range, the end is exclusive, so k-1 is used for k.
    r = N - 1 - k
    for c in range(N - 1 - k - 1, k - 1, -1):
        grid[r][c] = num
        num += 1

    # Fill the left edge of the current layer (moving up)
    # The column is fixed at k. Rows go from N-1-k-1 down to k+1.
    c = k
    for r in range(N - 1 - k - 1, k, -1):
        grid[r][c] = num
        num += 1

# Place 'T' at the center cell after all numbers are placed
grid[mid][mid] = 'T'

# Print the grid
for row in grid:
    # Convert each element in the row to a string before joining
    # This handles both numbers and 'T' correctly
    print(*[str(cell) for cell in row])