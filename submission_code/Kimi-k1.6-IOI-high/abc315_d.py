import sys
from collections import defaultdict

def main():
    H, W = map(int, sys.stdin.readline().split())
    c = [sys.stdin.readline().strip() for _ in range(H)]
    
    marked_rows = set()
    marked_columns = set()
    
    # Initialize row color counts and totals
    color_count_row = [defaultdict(int) for _ in range(H)]
    total_unmarked_columns_row = [W] * H
    for i in range(H):
        for j in range(W):
            color = c[i][j]
            color_count_row[i][color] += 1
    
    # Initialize column color counts and totals
    color_count_col = [defaultdict(int) for _ in range(W)]
    total_unmarked_rows_col = [H] * W
    for j in range(W):
        for i in range(H):
            color = c[i][j]
            color_count_col[j][color] += 1
    
    while True:
        new_rows = set()
        new_cols = set()
        
        # Check rows for eligibility
        for i in range(H):
            if i not in marked_rows and total_unmarked_columns_row[i] >= 2:
                if len(color_count_row[i]) == 1:
                    new_rows.add(i)
        
        # Check columns for eligibility
        for j in range(W):
            if j not in marked_columns and total_unmarked_rows_col[j] >= 2:
                if len(color_count_col[j]) == 1:
                    new_cols.add(j)
        
        if not new_rows and not new_cols:
            break
        
        # Process new_rows and update columns
        for i in new_rows:
            marked_rows.add(i)
            for j in range(W):
                if j not in marked_columns:
                    color = c[i][j]
                    color_count_col[j][color] -= 1
                    if color_count_col[j][color] == 0:
                        del color_count_col[j][color]
                    total_unmarked_rows_col[j] -= 1
        
        # Process new_cols and update rows
        for j in new_cols:
            marked_columns.add(j)
            for i in range(H):
                if i not in marked_rows:
                    color = c[i][j]
                    color_count_row[i][color] -= 1
                    if color_count_row[i][color] == 0:
                        del color_count_row[i][color]
                    total_unmarked_columns_row[i] -= 1
    
    remaining_rows = H - len(marked_rows)
    remaining_cols = W - len(marked_columns)
    print(remaining_rows * remaining_cols)

if __name__ == "__main__":
    main()