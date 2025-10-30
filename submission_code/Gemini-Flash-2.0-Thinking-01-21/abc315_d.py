def solve():
    h, w = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    
    while True:
        marked_positions = set()
        removed_in_iteration = False
        
        # Row operations
        for i in range(h):
            row_colors = []
            present_indices = []
            for j in range(w):
                if grid[i][j] != '.':
                    row_colors.append(grid[i][j])
                    present_indices.append(j)
            if len(row_colors) >= 2 and len(set(row_colors)) == 1:
                for j_index in present_indices:
                    marked_positions.add((i, j_index))
                    
        # Column operations
        for j in range(w):
            col_colors = []
            present_indices = []
            for i in range(h):
                if grid[i][j] != '.':
                    col_colors.append(grid[i][j])
                    present_indices.append(i)
            if len(col_colors) >= 2 and len(set(col_colors)) == 1:
                for i_index in present_indices:
                    marked_positions.add((i_index, j))
                    
        if not marked_positions:
            break
        else:
            removed_in_iteration = True
            for r, c in marked_positions:
                grid[r][c] = '.'
                
        if not removed_in_iteration:
            break
            
    remaining_cookies_count = 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] != '.':
                remaining_cookies_count += 1
                
    print(remaining_cookies_count)

if __name__ == '__main__':
    solve()