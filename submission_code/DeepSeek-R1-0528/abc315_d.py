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
    
    active_row = [set(range(W)) for _ in range(H)]
    active_col = [set(range(H)) for _ in range(W)]
    
    row_count = [W] * H
    col_count = [H] * W
    
    row_dict = [defaultdict(int) for _ in range(H)]
    col_dict = [defaultdict(int) for _ in range(W)]
    
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            row_dict[i][c] += 1
            col_dict[j][c] += 1

    total_remaining = H * W
    
    rows_to_check = set(range(H))
    cols_to_check = set(range(W))
    
    while rows_to_check or cols_to_check:
        rows_to_remove = []
        for i in rows_to_check:
            if row_count[i] >= 2 and len(row_dict[i]) == 1:
                rows_to_remove.append(i)
                
        cols_to_remove = []
        for j in cols_to_check:
            if col_count[j] >= 2 and len(col_dict[j]) == 1:
                cols_to_remove.append(j)
                
        if not rows_to_remove and not cols_to_remove:
            break
            
        removed_set = set()
        
        for i in rows_to_remove:
            for j in active_row[i]:
                removed_set.add((i, j))
                
        for j in cols_to_remove:
            for i in active_col[j]:
                removed_set.add((i, j))
                
        total_remaining -= len(removed_set)
        
        next_rows_to_check = set()
        next_cols_to_check = set()
        
        for (i, j) in removed_set:
            c = grid[i][j]
            
            if row_dict[i][c] > 0:
                row_dict[i][c] -= 1
                if row_dict[i][c] == 0:
                    del row_dict[i][c]
            row_count[i] -= 1
            
            if col_dict[j][c] > 0:
                col_dict[j][c] -= 1
                if col_dict[j][c] == 0:
                    del col_dict[j][c]
            col_count[j] -= 1
            
            active_row[i].discard(j)
            active_col[j].discard(i)
            
            next_rows_to_check.add(i)
            next_cols_to_check.add(j)
            
        rows_to_check = next_rows_to_check
        cols_to_check = next_cols_to_check
        
    print(total_remaining)

if __name__ == '__main__':
    main()