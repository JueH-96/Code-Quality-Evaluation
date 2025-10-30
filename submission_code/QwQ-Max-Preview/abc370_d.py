import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    # Initialize rows and columns
    rows = [[] for _ in range(H + 1)]  # rows[r] contains sorted columns with walls
    cols = [[] for _ in range(W + 1)]  # cols[c] contains sorted rows with walls
    
    for r in range(1, H + 1):
        rows[r] = list(range(1, W + 1))
    for c in range(1, W + 1):
        cols[c] = list(range(1, H + 1))
    
    for _ in range(Q):
        Rq = int(data[idx])
        idx += 1
        Cq = int(data[idx])
        idx += 1
        
        # Check if the current cell is a wall
        row_cols = rows[Rq]
        idx_row = bisect.bisect_left(row_cols, Cq)
        present_row = idx_row < len(row_cols) and row_cols[idx_row] == Cq
        
        col_rows = cols[Cq]
        idx_col = bisect.bisect_left(col_rows, Rq)
        present_col = idx_col < len(col_rows) and col_rows[idx_col] == Rq
        
        if present_row and present_col:
            # Remove the wall from rows and columns
            row_cols.pop(idx_row)
            col_rows.pop(idx_col)
        else:
            to_destroy = []
            
            # Up direction: find largest row < Rq in cols[Cq]
            col_rows = cols[Cq]
            idx = bisect.bisect_left(col_rows, Rq)
            if idx > 0:
                i = col_rows[idx - 1]
                to_destroy.append((i, Cq))
            
            # Down direction: find smallest row > Rq in cols[Cq]
            col_rows = cols[Cq]
            idx = bisect.bisect_left(col_rows, Rq)
            if idx < len(col_rows) and col_rows[idx] > Rq:
                i = col_rows[idx]
                to_destroy.append((i, Cq))
            
            # Left direction: find largest column < Cq in rows[Rq]
            row_cols = rows[Rq]
            idx = bisect.bisect_left(row_cols, Cq)
            if idx > 0:
                j = row_cols[idx - 1]
                to_destroy.append((Rq, j))
            
            # Right direction: find smallest column > Cq in rows[Rq]
            row_cols = rows[Rq]
            idx = bisect.bisect_left(row_cols, Cq)
            if idx < len(row_cols) and row_cols[idx] > Cq:
                j = row_cols[idx]
                to_destroy.append((Rq, j))
            
            # Destroy all collected walls
            for (i, j) in to_destroy:
                # Remove j from row i's columns list
                row_list = rows[i]
                jdx = bisect.bisect_left(row_list, j)
                if jdx < len(row_list) and row_list[jdx] == j:
                    row_list.pop(jdx)
                
                # Remove i from column j's rows list
                col_list = cols[j]
                idx_col = bisect.bisect_left(col_list, i)
                if idx_col < len(col_list) and col_list[idx_col] == i:
                    col_list.pop(idx_col)
    
    # Calculate the total remaining walls
    total = 0
    for r in range(1, H + 1):
        total += len(rows[r])
    print(total)

if __name__ == "__main__":
    main()