H, W, M = map(int, input().split())

operations = []
for _ in range(M):
    t, a, x = map(int, input().split())
    operations.append((t, a, x))

# Track the last operation that affected each row and column
last_row_op = [-1] * (H + 1)  # last_row_op[i] = operation index that last affected row i
last_col_op = [-1] * (W + 1)  # last_col_op[j] = operation index that last affected column j

# Process operations to find the last one affecting each row/column
for i, (t, a, x) in enumerate(operations):
    if t == 1:  # row operation
        last_row_op[a] = i
    else:  # column operation
        last_col_op[a] = i

# Count colors
color_count = {}

# For each cell (i,j), determine its final color
for i in range(1, H + 1):
    for j in range(1, W + 1):
        # Find the last operation that affected this cell
        row_op_idx = last_row_op[i]
        col_op_idx = last_col_op[j]
        
        if row_op_idx == -1 and col_op_idx == -1:
            # No operations affected this cell, it remains color 0
            final_color = 0
        elif row_op_idx == -1:
            # Only column operation affected this cell
            final_color = operations[col_op_idx][2]
        elif col_op_idx == -1:
            # Only row operation affected this cell
            final_color = operations[row_op_idx][2]
        else:
            # Both row and column operations affected this cell
            # The later operation determines the final color
            if row_op_idx > col_op_idx:
                final_color = operations[row_op_idx][2]
            else:
                final_color = operations[col_op_idx][2]
        
        color_count[final_color] = color_count.get(final_color, 0) + 1

# Output results
colors = sorted(color_count.keys())
print(len(colors))
for color in colors:
    print(color, color_count[color])