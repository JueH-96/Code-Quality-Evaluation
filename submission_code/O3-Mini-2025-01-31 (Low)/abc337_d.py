def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    K = int(next(it))
    grid = [list(next(it).strip()) for _ in range(H)]
    
    INF = 10**9
    ans = INF

    # Check horizontal segments if K is not longer than the number of columns.
    if K <= W:
        for i in range(H):
            row = grid[i]
            # sliding window in row: window [j, j+K-1]
            # maintain two counters
            #   'ops' counts number of '.' in window (each requires one operation)
            #   'bad' counts number of 'x' in window (cells that cannot be changed)
            ops = 0
            bad = 0
            # initialize the first window in row i
            for j in range(K):
                if row[j] == '.':
                    ops += 1
                elif row[j] == 'x':
                    bad += 1
            if bad == 0:
                ans = min(ans, ops)
            # slide the window across the row
            for j in range(K, W):
                # remove element leaving window (at index j-K)
                if row[j-K] == '.':
                    ops -= 1
                elif row[j-K] == 'x':
                    bad -= 1
                # add element entering the window at index j
                if row[j] == '.':
                    ops += 1
                elif row[j] == 'x':
                    bad += 1
                if bad == 0:
                    ans = min(ans, ops)
    
    # Check vertical segments if K is not longer than the number of rows.
    if K <= H:
        for j in range(W):
            ops = 0
            bad = 0
            # initialize first vertical window in column j: rows 0..K-1
            for i in range(K):
                cell = grid[i][j]
                if cell == '.':
                    ops += 1
                elif cell == 'x':
                    bad += 1
            if bad == 0:
                ans = min(ans, ops)
            # slide down the column
            for i in range(K, H):
                # remove element leaving window at row i-K
                if grid[i-K][j] == '.':
                    ops -= 1
                elif grid[i-K][j] == 'x':
                    bad -= 1
                # add element entering window at row i
                if grid[i][j] == '.':
                    ops += 1
                elif grid[i][j] == 'x':
                    bad += 1
                if bad == 0:
                    ans = min(ans, ops)

    # If no valid segment found, output -1.
    if ans == INF:
        sys.stdout.write(str(-1))
    else:
        sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()