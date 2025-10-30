H, W, M = map(int, input().split())

# Initialize grid with zeros
grid = [[0] * W for _ in range(H)]

# Keep track of last operation on each row and column
row_ops = [(-1, 0)] * H  # (time, color)
col_ops = [(-1, 0)] * W  # (time, color)

# Process operations
for i in range(M):
    t, a, x = map(int, input().split())
    a -= 1  # Convert to 0-based indexing
    
    if t == 1:
        row_ops[a] = (i, x)
    else:
        col_ops[a] = (i, x)

# Count colors
color_count = {}

# Process columns first
for j in range(W):
    col_time, col_color = col_ops[j]
    count = 0
    
    for i in range(H):
        row_time, row_color = row_ops[i]
        
        # Cell color is determined by the latest operation
        if row_time > col_time:
            final_color = row_color
        else:
            final_color = col_color
            
        if final_color not in color_count:
            color_count[final_color] = 0
        color_count[final_color] += 1

# Print results
print(len(color_count))
for color in sorted(color_count.keys()):
    print(color, color_count[color])