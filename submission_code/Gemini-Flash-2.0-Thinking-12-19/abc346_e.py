import collections

def solve():
    h, w, m = map(int, input().split())
    operations = []
    for _ in range(m):
        operations.append(list(map(int, input().split())))
    
    last_row_op_index = [-1] * h
    last_col_op_index = [-1] * w
    row_color = [0] * h
    col_color = [0] * w
    
    for i in range(m):
        op_type, index, color = operations[i]
        if op_type == 1:
            last_row_op_index[index-1] = i + 1
            row_color[index-1] = color
        elif op_type == 2:
            last_col_op_index[index-1] = i + 1
            col_color[index-1] = color
            
    grid_colors = []
    for r in range(h):
        row_colors = []
        for c in range(w):
            row_op_idx = last_row_op_index[r]
            col_op_idx = last_col_op_index[c]
            cell_color = 0
            if row_op_idx != -1 and col_op_idx == -1:
                cell_color = row_color[r]
            elif row_op_idx == -1 and col_op_idx != -1:
                cell_color = col_color[c]
            elif row_op_idx != -1 and col_op_idx != -1:
                if row_op_idx > col_op_idx:
                    cell_color = row_color[r]
                else:
                    cell_color = col_color[c]
            else:
                cell_color = 0
            row_colors.append(cell_color)
        grid_colors.append(row_colors)
        
    color_counts = collections.defaultdict(int)
    for i in range(h):
        for j in range(w):
            color_counts[grid_colors[i][j]] += 1
            
    distinct_colors = sorted([color for color in color_counts if color_counts[color] > 0])
    print(len(distinct_colors))
    for color in distinct_colors:
        print(color, color_counts[color])

if __name__ == '__main__':
    solve()