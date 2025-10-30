from collections import defaultdict

H, W, M = map(int, input().split())

# Track last operation for each row and column
# -1 means never painted, otherwise it's the operation index
row_op = [-1] * H
row_color = [0] * H
col_op = [-1] * W
col_color = [0] * W

# Process operations
for op_idx in range(M):
    T, A, X = map(int, input().split())
    if T == 1:  # Row operation
        row_op[A-1] = op_idx
        row_color[A-1] = X
    else:  # Column operation
        col_op[A-1] = op_idx
        col_color[A-1] = X

# Count cells by color
color_count = defaultdict(int)

for i in range(H):
    for j in range(W):
        if row_op[i] > col_op[j]:
            color = row_color[i]
        else:
            color = col_color[j]
        
        color_count[color] += 1

# Output results
colors = sorted(color_count.keys())
print(len(colors))
for color in colors:
    print(color, color_count[color])