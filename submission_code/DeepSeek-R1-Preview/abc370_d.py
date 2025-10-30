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
    
    walls = set()
    rows = dict()
    cols = dict()
    
    # Initialize rows
    for R in range(1, H+1):
        cols_in_row = list(range(1, W+1))
        rows[R] = cols_in_row
        for C in cols_in_row:
            walls.add((R, C))
    
    # Initialize columns
    for C in range(1, W+1):
        rows_in_col = list(range(1, H+1))
        cols[C] = rows_in_col
    
    for _ in range(Q):
        R = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        
        if (R, C) in walls:
            # Remove (R, C) from walls
            walls.remove((R, C))
            # Remove from rows[R]
            row = rows[R]
            pos = bisect.bisect_left(row, C)
            if pos < len(row) and row[pos] == C:
                del row[pos]
            # Remove from cols[C]
            col = cols[C]
            pos_col = bisect.bisect_left(col, R)
            if pos_col < len(col) and col[pos_col] == R:
                del col[pos_col]
        else:
            to_destroy = set()
            # Check left
            if R in rows:
                row = rows[R]
                pos = bisect.bisect_left(row, C)
                if pos > 0:
                    left = row[pos - 1]
                    to_destroy.add((R, left))
            # Check right
            if R in rows:
                row = rows[R]
                pos = bisect.bisect_right(row, C)
                if pos < len(row):
                    right = row[pos]
                    if right > C:
                        to_destroy.add((R, right))
            # Check up
            if C in cols:
                col = cols[C]
                pos = bisect.bisect_left(col, R)
                if pos > 0:
                    up = col[pos - 1]
                    to_destroy.add((up, C))
            # Check down
            if C in cols:
                col = cols[C]
                pos = bisect.bisect_right(col, R)
                if pos < len(col):
                    down = col[pos]
                    if down > R:
                        to_destroy.add((down, C))
            # Process to_destroy
            for (i, j) in to_destroy:
                if (i, j) not in walls:
                    continue
                walls.remove((i, j))
                # Remove from rows[i]
                if i in rows:
                    row_i = rows[i]
                    pos_i = bisect.bisect_left(row_i, j)
                    if pos_i < len(row_i) and row_i[pos_i] == j:
                        del row_i[pos_i]
                # Remove from cols[j]
                if j in cols:
                    col_j = cols[j]
                    pos_j = bisect.bisect_left(col_j, i)
                    if pos_j < len(col_j) and col_j[pos_j] == i:
                        del col_j[pos_j]
    
    print(len(walls))

if __name__ == "__main__":
    main()