# YOUR CODE HERE
H, W, M = map(int, input().split())

# Store last operation for each row and column
# Format: (operation_time, color)
row_ops = {}  # row_number -> (time, color)
col_ops = {}  # col_number -> (time, color)

# Process operations
for i in range(M):
    T, A, X = map(int, input().split())
    if T == 1:  # Row operation
        row_ops[A-1] = (i, X)  # 0-indexed
    else:  # Column operation
        col_ops[A-1] = (i, X)  # 0-indexed

# Count cells by color
color_count = {}

# For each cell, determine its final color
for r in range(H):
    for c in range(W):
        # Check last operations on this row and column
        row_time = row_ops.get(r, (-1, 0))[0]
        row_color = row_ops.get(r, (-1, 0))[1]
        
        col_time = col_ops.get(c, (-1, 0))[0]
        col_color = col_ops.get(c, (-1, 0))[1]
        
        # The cell's color is determined by the latest operation
        if row_time > col_time:
            final_color = row_color
        elif col_time > row_time:
            final_color = col_color
        else:  # No operations on this row or column
            final_color = 0
        
        # Count this color
        if final_color not in color_count:
            color_count[final_color] = 0
        color_count[final_color] += 1

# Output results
colors = sorted(color_count.keys())
print(len(colors))
for color in colors:
    print(color, color_count[color])