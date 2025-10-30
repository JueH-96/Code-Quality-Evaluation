import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
grid = []
for _ in range(N):
    row_str = data[index]
    index += 1
    grid.append(row_str)

# Now for each output row and column (1-based)
for out_row in range(1, N + 1):
    row_output = []
    for out_col in range(1, N + 1):
        # Compute k
        k = min(out_row, out_col, N + 1 - out_row, N + 1 - out_col)
        # Compute m = k mod 4
        m = k % 4
        # Compute source position based on m
        if m == 0:
            src_row = out_row
            src_col = out_col
        elif m == 1:
            src_row = N + 1 - out_col
            src_col = out_row
        elif m == 2:
            src_row = N + 1 - out_row
            src_col = N + 1 - out_col
        elif m == 3:
            src_row = out_col
            src_col = N + 1 - out_row
        # Get the color from initial grid (adjust to 0-based indexing)
        color = grid[src_row - 1][src_col - 1]
        row_output.append(color)
    # Print the row as a string
    print(''.join(row_output))