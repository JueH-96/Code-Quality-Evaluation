import sys

# Read N from standard input
N = int(sys.stdin.readline())

# Calculate the maximum 1-indexed 'i' for which an operation is performed.
# An operation for index 'i' is performed only if i <= N + 1 - i,
# which simplifies to 2 * i <= N + 1.
# This means i <= (N + 1) / 2. Since 'i' must be an integer,
# the maximum value of 'i' for which an operation is performed is floor((N + 1) / 2).
i_limit = (N + 1) // 2

# Initialize an N x N grid represented as a list of lists.
# We'll store characters ('#' or '.') directly in the grid.
grid = [['' for _ in range(N)] for _ in range(N)]

# Iterate through each cell (r, c) in the grid, where r is the row index
# (0 to N-1) and c is the column index (0 to N-1).
for r in range(N):
    for c in range(N):
        # For a cell at 0-indexed position (r, c), its 1-indexed position is (r+1, c+1).

        # Calculate the minimum 1-indexed distance from the cell's row 'r'
        # to the nearest horizontal boundary (top row 0 or bottom row N-1).
        # Distance to top boundary (row 0): r + 1
        # Distance to bottom boundary (row N-1): (N - 1 - r) + 1 = N - r
        dist_r = min(r + 1, N - r)

        # Calculate the minimum 1-indexed distance from the cell's column 'c'
        # to the nearest vertical boundary (left column 0 or right column N-1).
        # Distance to left boundary (col 0): c + 1
        # Distance to right boundary (col N-1): (N - 1 - c) + 1 = N - c
        dist_c = min(c + 1, N - c)

        # 'm' represents the largest 1-indexed 'i' such that the cell (r, c)
        # is geometrically contained within the boundaries of the rectangle
        # defined by operation 'i'. The i-th rectangle (1-indexed) spans
        # rows from 'i' to 'N+1-i' and columns from 'i' to 'N+1-i'.
        # A cell (r+1, c+1) is within this rectangle if:
        # i <= r+1 <= N+1-i  AND  i <= c+1 <= N+1-i
        # These inequalities together imply:
        # i <= r+1
        # i <= N+1 - (r+1) = N - r
        # i <= c+1
        # i <= N+1 - (c+1) = N - c
        # So, cell (r, c) is in rectangle 'i' if i <= min(r+1, N-r, c+1, N-c).
        m = min(dist_r, dist_c)

        # 'k' is the largest 1-indexed 'i' such that operation 'i' actually
        # colors cell (r, c). This requires two conditions to be met simultaneously:
        # 1. The cell must be geometrically inside the i-th rectangle (i <= m).
        # 2. Operation 'i' must be performed (i <= i_limit).
        # The operations are performed for i = 1, 2, ..., i_limit.
        # The cell (r,c) is colored by operation i if 1 <= i <= m AND 1 <= i <= i_limit.
        # The largest such 'i' is k = min(m, i_limit). Since operations overwrite
        # previous ones, the final color is determined by the color applied by
        # the operation with index 'k'.
        k = min(m, i_limit)

        # The color applied by operation 'i' is black ('#') if 'i' is odd,
        # and white ('.') if 'i' is even.
        # Therefore, the final color of the cell (r, c) is '#' if 'k' is odd,
        # and '.' if 'k' is even.
        if k % 2 == 1:
            grid[r][c] = '#'
        else:
            grid[r][c] = '.'

# Print the resulting grid row by row.
for r in range(N):
    # Join the characters in the row list into a single string and print it.
    print("".join(grid[r]))