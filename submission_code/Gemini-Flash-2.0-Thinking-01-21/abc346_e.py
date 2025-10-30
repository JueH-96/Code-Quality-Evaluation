import collections

def solve():
    h, w, m = map(int, input().split())
    operations = []
    for _ in range(m):
        operations.append(list(map(int, input().split())))
    
    last_row_op_index = [0] * h
    last_col_op_index = [0] * w
    op_colors = []
    
    for i in range(m):
        op_type, index, color = operations[i]
        op_colors.append(color)
        if op_type == 1:
            last_row_op_index[index-1] = i + 1
        elif op_type == 2:
            last_col_op_index[index-1] = i + 1
            
    color_counts = collections.defaultdict(int)
    
    for r in range(h):
        for c in range(w):
            row_op_idx = last_row_op_index[r]
            col_op_idx = last_col_op_index[c]
            cell_color = 0
            if row_op_idx == 0 and col_op_idx == 0:
                cell_color = 0
            elif row_op_idx > col_op_idx:
                cell_color = op_colors[row_op_idx-1]
            else:
                cell_color = op_colors[col_op_idx-1]
            color_counts[cell_color] += 1
            
    distinct_colors = sorted([color for color in color_counts if color_counts[color] > 0])
    print(len(distinct_colors))
    for color in distinct_colors:
        print(f"{color} {color_counts[color]}")

if __name__ == '__main__':
    solve()