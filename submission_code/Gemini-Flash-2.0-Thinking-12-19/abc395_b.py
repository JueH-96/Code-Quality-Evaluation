import sys

# Read N from standard input
N = int(sys.stdin.readline())

# Calculate (N+1)//2 once. This is the maximum index `i` for which an operation is performed.
# The operation for index `i` is performed only if i <= N + 1 - i,
# which simplifies to 2*i <= N + 1, or i <= (N+1)//2 for integer i.
max_op_index = (N + 1) // 2

# Create an N x N grid represented as a list of lists of characters.
grid = [['' for _ in range(N)] for _ in range(N)]

# Iterate through each cell (row, col) using 0-indexed coordinates.
# row: 0 to N-1, col: 0 to N-1.
# The 1-indexed row is r = row + 1
# The 1-indexed column is c = col + 1
for row in range(N):
    for col in range(N):
        # An operation with index 'i' (1-indexed) colors the rectangle
        # from (i, i) to (N+1-i, N+1-i).
        # Cell (r, c) is covered by operation 'i' if i <= r <= N+1-i and i <= c <= N+1-i.
        # This is equivalent to:
        # i <= r AND i <= N+1-r AND i <= c AND i <= N+1-c.
        # So, cell (r, c) is covered by operation 'i' if 1 <= i <= min(r, N+1-r, c, N+1-c).
        # Let k = min(r, N+1-r, c, N+1-c).
        # Using 0-indexed coordinates for row and col:
        # r = row + 1
        # N+1-r = N+1-(row+1) = N-row
        # c = col + 1
        # N+1-c = N+1-(col+1) = N-col
        # So, k = min(row + 1, N - row, col + 1, N - col).
        k = min(row + 1, N - row, col + 1, N - col)

        # The operations are performed in increasing order of 'i', but only up to max_op_index.
        # The final color of cell (r, c) is determined by the last operation that covers it.
        # This is the operation with the largest index 'i' such that 1 <= i <= k AND 1 <= i <= max_op_index.
        # The largest such index is i_final = min(k, max_op_index).
        i_final = min(k, max_op_index)

        # The color applied by operation 'i' is '#' if i is odd, '.' if i is even.
        # So, the final color depends on the parity of i_final.
        if i_final % 2 == 1:
            grid[row][col] = '#'
        else:
            grid[row][col] = '.'

# Print the grid row by row.
for row in grid:
    print("".join(row))