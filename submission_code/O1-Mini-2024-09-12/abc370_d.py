import sys, bisect

def main():
    import sys, bisect
    input = sys.stdin.read().split()
    H = int(input[0])
    W = int(input[1])
    Q = int(input[2])
    queries = list(map(int, input[3:]))
    
    row_walls = [list(range(1, W+1)) for _ in range(H)]
    col_walls = [list(range(1, H+1)) for _ in range(W)]
    total_walls = H * W
    
    for q in range(Q):
        R = queries[2*q]
        C = queries[2*q+1]
        row = row_walls[R-1]
        idx = bisect.bisect_left(row, C)
        if idx < len(row) and row[idx] == C:
            # Wall exists, destroy it
            del row[idx]
            col = col_walls[C-1]
            idx_col = bisect.bisect_left(col, R)
            if idx_col < len(col) and col[idx_col] == R:
                del col[idx_col]
            total_walls -=1
        else:
            to_remove = set()
            # Up
            col = col_walls[C-1]
            idx_up = bisect.bisect_left(col, R)
            if idx_up > 0:
                i = col[idx_up-1]
                to_remove.add( (i, C) )
            # Down
            idx_down = bisect.bisect_right(col, R)
            if idx_down < len(col):
                i = col[idx_down]
                to_remove.add( (i, C) )
            # Left
            row = row_walls[R-1]
            idx_left = bisect.bisect_left(row, C)
            if idx_left >0:
                j = row[idx_left-1]
                to_remove.add( (R, j) )
            # Right
            idx_right = bisect.bisect_right(row, C)
            if idx_right < len(row):
                j = row[idx_right]
                to_remove.add( (R, j) )
            # Remove walls
            for (i,j) in to_remove:
                # Remove j from row_walls[i-1]
                row_i = row_walls[i-1]
                idx_row_i = bisect.bisect_left(row_i, j)
                if idx_row_i < len(row_i) and row_i[idx_row_i] == j:
                    del row_i[idx_row_i]
                # Remove i from col_walls[j-1]
                col_j = col_walls[j-1]
                idx_col_j = bisect.bisect_left(col_j, i)
                if idx_col_j < len(col_j) and col_j[idx_col_j] == i:
                    del col_j[idx_col_j]
                total_walls -=1
    print(total_walls)

if __name__ == '__main__':
    main()