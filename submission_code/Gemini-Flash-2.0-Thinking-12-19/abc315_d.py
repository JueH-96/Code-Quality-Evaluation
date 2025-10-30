def solve():
    h, w = map(int, input().split())
    colors = [list(input()) for _ in range(h)]
    is_present = [[True for _ in range(w)] for _ in range(h)]
    
    while True:
        marked = [[False for _ in range(w)] for _ in range(h)]
        marked_in_iteration = False
        
        # Row-wise marking
        for i in range(h):
            present_cols = []
            row_colors = []
            for j in range(w):
                if is_present[i][j]:
                    present_cols.append(j)
                    row_colors.append(colors[i][j])
            if len(row_colors) >= 2:
                first_color = row_colors[0]
                all_same_color = True
                for color in row_colors:
                    if color != first_color:
                        all_same_color = False
                        break
                if all_same_color:
                    for j in present_cols:
                        marked[i][j] = True
                    if present_cols:
                        marked_in_iteration = True
                        
        # Column-wise marking
        for j in range(w):
            present_rows = []
            col_colors = []
            for i in range(h):
                if is_present[i][j]:
                    present_rows.append(i)
                    col_colors.append(colors[i][j])
            if len(col_colors) >= 2:
                first_color = col_colors[0]
                all_same_color = True
                for color in col_colors:
                    if color != first_color:
                        all_same_color = False
                        break
                if all_same_color:
                    for i in present_rows:
                        marked[i][j] = True
                    if present_rows:
                        marked_in_iteration = True
                        
        if not marked_in_iteration:
            break
            
        for i in range(h):
            for j in range(w):
                if marked[i][j]:
                    is_present[i][j] = False
                    
    remaining_cookies_count = 0
    for i in range(h):
        for j in range(w):
            if is_present[i][j]:
                remaining_cookies_count += 1
                
    print(remaining_cookies_count)

if __name__ == '__main__':
    solve()