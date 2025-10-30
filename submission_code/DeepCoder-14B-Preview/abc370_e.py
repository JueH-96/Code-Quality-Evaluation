import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    ptr = 0
    H = int(data[ptr])
    ptr += 1
    W = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    
    grid = [[1] * W for _ in range(H)]
    
    walls_in_column = [[] for _ in range(W + 1)]  # 1-based indexing for columns
    for c in range(1, W + 1):
        walls_in_column[c] = list(range(1, H + 1))
    
    walls_in_row = [[] for _ in range(H + 1)]  # 1-based indexing for rows
    for r in range(1, H + 1):
        walls_in_row[r] = list(range(1, W + 1))
    
    for _ in range(Q):
        R = int(data[ptr])
        ptr += 1
        C = int(data[ptr])
        ptr += 1
        
        if grid[R-1][C-1] == 1:
            grid[R-1][C-1] = 0
            
            # Remove R from walls_in_column[C]
            idx = bisect.bisect_left(walls_in_column[C], R)
            if idx < len(walls_in_column[C]) and walls_in_column[C][idx] == R:
                del walls_in_column[C][idx]
            
            # Remove C from walls_in_row[R]
            idx = bisect.bisect_left(walls_in_row[R], C)
            if idx < len(walls_in_row[R]) and walls_in_row[R][idx] == C:
                del walls_in_row[R][idx]
            continue
        
        # Process up direction
        c = C
        lst = walls_in_column[c]
        if lst:
            idx = bisect.bisect_left(lst, R) - 1
            if idx >= 0:
                i = lst[idx]
                left = bisect.bisect_right(lst, i)
                right = bisect.bisect_left(lst, R)
                if right - left == 0:
                    grid[i-1][c-1] = 0
                    
                    # Remove i from walls_in_column[c]
                    pos = bisect.bisect_left(lst, i)
                    if pos < len(lst) and lst[pos] == i:
                        del lst[pos]
                    
                    # Remove c from walls_in_row[i]
                    row_lst = walls_in_row[i]
                    pos_row = bisect.bisect_left(row_lst, c)
                    if pos_row < len(row_lst) and row_lst[pos_row] == c:
                        del row_lst[pos_row]
        
        # Process down direction
        c = C
        lst = walls_in_column[c]
        if lst:
            idx = bisect.bisect_right(lst, R)
            if idx < len(lst):
                i = lst[idx]
                grid[i-1][c-1] = 0
                
                # Remove i from walls_in_column[c]
                pos = bisect.bisect_left(lst, i)
                if pos < len(lst) and lst[pos] == i:
                    del lst[pos]
                
                # Remove c from walls_in_row[i]
                row_lst = walls_in_row[i]
                pos_row = bisect.bisect_left(row_lst, c)
                if pos_row < len(row_lst) and row_lst[pos_row] == c:
                    del row_lst[pos_row]
        
        # Process left direction
        r = R
        lst = walls_in_row[r]
        if lst:
            idx = bisect.bisect_left(lst, C) - 1
            if idx >= 0:
                j = lst[idx]
                left = bisect.bisect_right(lst, j)
                right = bisect.bisect_left(lst, C)
                if right - left == 0:
                    grid[r-1][j-1] = 0
                    
                    # Remove j from walls_in_row[r]
                    pos = bisect.bisect_left(lst, j)
                    if pos < len(lst) and lst[pos] == j:
                        del lst[pos]
                    
                    # Remove r from walls_in_column[j]
                    col_lst = walls_in_column[j]
                    pos_col = bisect.bisect_left(col_lst, r)
                    if pos_col < len(col_lst) and col_lst[pos_col] == r:
                        del col_lst[pos_col]
        
        # Process right direction
        r = R
        lst = walls_in_row[r]
        if lst:
            idx = bisect.bisect_right(lst, C)
            if idx < len(lst):
                j = lst[idx]
                grid[r-1][j-1] = 0
                
                # Remove j from walls_in_row[r]
                pos = bisect.bisect_left(lst, j)
                if pos < len(lst) and lst[pos] == j:
                    del lst[pos]
                
                # Remove r from walls_in_column[j]
                col_lst = walls_in_column[j]
                pos_col = bisect.bisect_left(col_lst, r)
                if pos_col < len(col_lst) and col_lst[pos_col] == r:
                    del col_lst[pos_col]
    
    # Calculate remaining walls
    remaining = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 1:
                remaining += 1
    print(remaining)

if __name__ == "__main__":
    main()