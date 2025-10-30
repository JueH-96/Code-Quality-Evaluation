H, W, M = map(int, input().split())

# Initialize arrays to keep track of the last color and operation time
row_color = [0] * (H + 1)  # +1 because rows are 1-indexed
row_time = [0] * (H + 1)   # Time of the last operation on each row
col_color = [0] * (W + 1)  # +1 because columns are 1-indexed
col_time = [0] * (W + 1)   # Time of the last operation on each column

# Process operations
for i in range(1, M + 1):
    T, A, X = map(int, input().split())
    if T == 1:  # Row operation
        row_color[A] = X
        row_time[A] = i
    else:  # Column operation
        col_color[A] = X
        col_time[A] = i

# Count cells of each color
color_count = {}

# For each row, count cells painted by row's color
for r in range(1, H + 1):
    row_colored_cells = 0
    for c in range(1, W + 1):
        # If row operation happened after column operation, the row color applies
        if row_time[r] > col_time[c]:
            row_colored_cells += 1
    if row_colored_cells > 0:
        color_count[row_color[r]] = color_count.get(row_color[r], 0) + row_colored_cells

# For each column, count cells painted by column's color
for c in range(1, W + 1):
    col_colored_cells = 0
    for r in range(1, H + 1):
        # If column operation happened at the same time or after row operation, the column color applies
        if col_time[c] >= row_time[r]:
            col_colored_cells += 1
    if col_colored_cells > 0:
        color_count[col_color[c]] = color_count.get(col_color[c], 0) + col_colored_cells

# Output
K = len(color_count)
print(K)
for color in sorted(color_count.keys()):
    print(color, color_count[color])