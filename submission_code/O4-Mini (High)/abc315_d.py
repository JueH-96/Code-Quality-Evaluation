import sys
def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    H = int(next(it)); W = int(next(it))
    # Read grid as bytes; grid[i][j] is ASCII code of character
    grid = [next(it) for _ in range(H)]
    base = 97  # ord('a')
    # Initialize degree and color‐count data structures
    cnt_row = [W] * H
    cnt_col = [H] * W
    unique_row = [0] * H
    unique_col = [0] * W
    colcnt_row = [[0]*26 for _ in range(H)]
    colcnt_col = [[0]*26 for _ in range(W)]
    # Build initial counts
    for i in range(H):
        row = grid[i]
        crow = colcnt_row[i]
        for j in range(W):
            k = row[j] - base
            if crow[k] == 0:
                unique_row[i] += 1
            crow[k] += 1
            ccol = colcnt_col[j]
            if ccol[k] == 0:
                unique_col[j] += 1
            ccol[k] += 1
    # Flags for removed rows/columns and in‐queue markers
    removed_row = [False] * H
    removed_col = [False] * W
    in_queue_row = [False] * H
    in_queue_col = [False] * W
    # Initialize first wave of eligible rows/columns
    curr_rows = []
    curr_cols = []
    for i in range(H):
        if cnt_row[i] >= 2 and unique_row[i] == 1:
            curr_rows.append(i)
            in_queue_row[i] = True
    for j in range(W):
        if cnt_col[j] >= 2 and unique_col[j] == 1:
            curr_cols.append(j)
            in_queue_col[j] = True
    # Total cookies remaining
    total_remain = H * W
    # Wave‐by‐wave removal
    while curr_rows or curr_cols:
        next_rows = []
        next_cols = []
        # Remove all rows in this wave
        for i in curr_rows:
            removed_row[i] = True
            row = grid[i]
            for j in range(W):
                if removed_col[j]:
                    continue
                # remove edge (i,j)
                k = row[j] - base
                cc = colcnt_col[j]
                cc[k] -= 1
                if cc[k] == 0:
                    unique_col[j] -= 1
                cnt_col[j] -= 1
                total_remain -= 1
                # if column j newly becomes eligible, schedule for next wave
                if (not removed_col[j] and not in_queue_col[j]
                        and cnt_col[j] >= 2 and unique_col[j] == 1):
                    in_queue_col[j] = True
                    next_cols.append(j)
        # Remove all columns in this wave
        for j in curr_cols:
            removed_col[j] = True
            for i in range(H):
                if removed_row[i]:
                    continue
                # remove edge (i,j)
                k = grid[i][j] - base
                rr = colcnt_row[i]
                rr[k] -= 1
                if rr[k] == 0:
                    unique_row[i] -= 1
                cnt_row[i] -= 1
                total_remain -= 1
                # if row i newly becomes eligible, schedule for next wave
                if (not removed_row[i] and not in_queue_row[i]
                        and cnt_row[i] >= 2 and unique_row[i] == 1):
                    in_queue_row[i] = True
                    next_rows.append(i)
        # advance to next wave
        curr_rows = next_rows
        curr_cols = next_cols
    # Output remaining cookies
    print(total_remain)

if __name__ == "__main__":
    main()