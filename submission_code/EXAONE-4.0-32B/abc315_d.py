import sys
from collections import defaultdict

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    row_count = [W] * H
    col_count = [H] * W
    
    row_color = [defaultdict(int) for _ in range(H)]
    col_color = [defaultdict(int) for _ in range(W)]
    
    alive_cols_in_row = [set(range(W)) for _ in range(H)]
    alive_rows_in_col = [set(range(H)) for _ in range(W)]
    
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            row_color[i][c] += 1
            col_color[j][c] += 1
            
    pending_rows = set(range(H))
    pending_cols = set(range(W))
    
    while pending_rows or pending_cols:
        current_round_bad_rows = set()
        current_round_bad_cols = set()
        
        for i in pending_rows:
            if row_count[i] >= 2 and len(row_color[i]) == 1:
                current_round_bad_rows.add(i)
                
        for j in pending_cols:
            if col_count[j] >= 2 and len(col_color[j]) == 1:
                current_round_bad_cols.add(j)
                
        pending_rows = set()
        pending_cols = set()
        
        if not current_round_bad_rows and not current_round_bad_cols:
            break
            
        removed_cells = set()
        for i in current_round_bad_rows:
            for j in list(alive_cols_in_row[i]):
                removed_cells.add((i, j))
                
        for j in current_round_bad_cols:
            for i in list(alive_rows_in_col[j]):
                removed_cells.add((i, j))
                
        for (i, j) in removed_cells:
            if j in alive_cols_in_row[i]:
                alive_cols_in_row[i].discard(j)
            if i in alive_rows_in_col[j]:
                alive_rows_in_col[j].discard(i)
                
            c = grid[i][j]
            row_count[i] -= 1
            col_count[j] -= 1
            
            if row_color[i][c] > 0:
                row_color[i][c] -= 1
                if row_color[i][c] == 0:
                    del row_color[i][c]
                    
            if col_color[j][c] > 0:
                col_color[j][c] -= 1
                if col_color[j][c] == 0:
                    del col_color[j][c]
                    
            pending_rows.add(i)
            pending_cols.add(j)
            
    total_remaining = sum(row_count)
    print(total_remaining)

if __name__ == "__main__":
    main()