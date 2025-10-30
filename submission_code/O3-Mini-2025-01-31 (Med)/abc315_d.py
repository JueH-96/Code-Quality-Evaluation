def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    grid = [next(it).strip() for _ in range(H)]
    
    # For each row and column we will maintain:
    # - row_count[i]: the number of “active” (i.e. not removed) cookies in row i.
    # - row_freq[i]: a list of 26 integers (for letters a-z) counting, among active columns, each letter.
    # Similarly for columns.
    active_rows = [True] * H
    active_cols = [True] * W
    row_count = [W] * H
    row_freq = [[0] * 26 for _ in range(H)]
    for i in range(H):
        for ch in grid[i]:
            row_freq[i][ord(ch) - 97] += 1

    col_count = [H] * W
    col_freq = [[0] * 26 for _ in range(W)]
    for j in range(W):
        for i in range(H):
            ch = grid[i][j]
            col_freq[j][ord(ch) - 97] += 1

    # The idea is to view the situation in terms of active rows and active columns.
    # A cookie at (i,j) is “remaining” if row i and column j are both active.
    # The removal procedure is:
    #   • Remove every row that (with at least 2 active cookies) is homogeneous (all cookies are the same).
    #   • Remove every column that (with at least 2 active cookies) is homogeneous.
    # Here removal means marking the row or column as inactive. 
    # When a row (or column) is removed, we update the frequency counts in all still active columns (or rows).
    #
    # We simulate until no row or column qualifies.
    
    while True:
        R_remove = []
        C_remove = []
        
        # Check each active row. A row qualifies if its number of active cookies (row_count[i]) >= 2 
        # and all these cookies are the same letter.
        for i in range(H):
            if not active_rows[i] or row_count[i] < 2:
                continue
            cnt = 0
            freq = row_freq[i]
            for k in range(26):
                if freq[k]:
                    cnt += 1
                    if cnt > 1:
                        break
            if cnt == 1:
                R_remove.append(i)
                
        # Similarly check each active column.
        for j in range(W):
            if not active_cols[j] or col_count[j] < 2:
                continue
            cnt = 0
            freq = col_freq[j]
            for k in range(26):
                if freq[k]:
                    cnt += 1
                    if cnt > 1:
                        break
            if cnt == 1:
                C_remove.append(j)
        
        # If no row or column qualifies, we are done.
        if not R_remove and not C_remove:
            break
        
        # Mark rows and columns removed in this iteration.
        for i in R_remove:
            active_rows[i] = False
        for j in C_remove:
            active_cols[j] = False
        
        # After removal, update the counts.
        # For each row removed, update the active columns by subtracting the cookie in that row.
        for i in R_remove:
            row_str = grid[i]
            for j in range(W):
                if active_cols[j]:
                    letter = ord(row_str[j]) - 97
                    col_count[j] -= 1
                    col_freq[j][letter] -= 1
                    
        # For each column removed, update the active rows by subtracting the cookie in that column.
        for j in C_remove:
            for i in range(H):
                if active_rows[i]:
                    letter = ord(grid[i][j]) - 97
                    row_count[i] -= 1
                    row_freq[i][letter] -= 1

    # The remaining cookies are those in active rows; row_count[i] tells us how many active columns remain for row i.
    remaining = 0
    for i in range(H):
        if active_rows[i]:
            remaining += row_count[i]
    sys.stdout.write(str(remaining))

if __name__ == '__main__':
    main()