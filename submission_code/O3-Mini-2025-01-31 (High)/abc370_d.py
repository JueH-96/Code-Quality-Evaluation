def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        H = int(next(it))
        W = int(next(it))
        Q = int(next(it))
    except StopIteration:
        return

    # Build grid: rows 1..H and columns 1..W.
    # grid[r][c] is True if a wall is present.
    grid = [None] * (H+1)
    for r in range(1, H+1):
        grid[r] = [False] + [True] * W  # index 0 unused

    # Build a Fenwick tree for each row.
    # rowFenw[r] is a list (1-indexed positions 1..W).
    rowFenw = [None]*(H+1)
    for r in range(1, H+1):
        fw = [0]*(W+1)
        for i in range(1, W+1):
            fw[i] = 1  # initially a wall exists in every cell
        for i in range(1, W+1):
            j = i + (i & -i)
            if j <= W:
                fw[j] += fw[i]
        rowFenw[r] = fw

    # Build a Fenwick tree for each column.
    colFenw = [None]*(W+1)
    for c in range(1, W+1):
        fw = [0]*(H+1)
        for i in range(1, H+1):
            fw[i] = 1
        for i in range(1, H+1):
            j = i + (i & -i)
            if j <= H:
                fw[j] += fw[i]
        colFenw[c] = fw

    # Fenw tree utility functions.
    # Get prefix sum (from 1 to i) in fenw array f.
    def fenw_sum(f, i):
        s = 0
        while i:
            s += f[i]
            i -= i & -i
        return s

    # Update fenw tree f (of size n) at position i by delta.
    def fenw_update(f, n, i, delta):
        while i <= n:
            f[i] += delta
            i += i & -i

    # Find smallest index i in [1,n] such that prefix sum >= target.
    def fenw_find(f, n, target):
        i = 0
        bit = 1 << (n.bit_length() - 1)
        while bit:
            ni = i + bit
            if ni <= n and f[ni] < target:
                target -= f[ni]
                i = ni
            bit //= 2
        return i+1

    # For speed, bind functions to local names.
    fs = fenw_sum
    fu = fenw_update
    ff = fenw_find

    removed = 0  # count of removed walls
    for _ in range(Q):
        try:
            r = int(next(it))
            c = int(next(it))
        except StopIteration:
            break
        if grid[r][c]:
            # Case 1: bomb falls on a cell with a wall.
            grid[r][c] = False
            removed += 1
            fu(rowFenw[r], W, c, -1)
            fu(colFenw[c], H, r, -1)
        else:
            # Case 2: bomb cell is empty.
            # We look in the 4 directions.
            up_r = None
            if r > 1:
                cnt = fs(colFenw[c], r-1)
                if cnt > 0:
                    up_r = ff(colFenw[c], H, cnt)
            down_r = None
            if r < H:
                prefix = fs(colFenw[c], r)
                total = fs(colFenw[c], H)
                if total > prefix:
                    down_r = ff(colFenw[c], H, prefix+1)
            left_c = None
            if c > 1:
                cnt = fs(rowFenw[r], c-1)
                if cnt > 0:
                    left_c = ff(rowFenw[r], W, cnt)
            right_c = None
            if c < W:
                prefix = fs(rowFenw[r], c)
                total = fs(rowFenw[r], W)
                if total > prefix:
                    right_c = ff(rowFenw[r], W, prefix+1)
            # Remove each of the found walls (if still there).
            if up_r is not None:
                if grid[up_r][c]:
                    grid[up_r][c] = False
                    removed += 1
                    fu(rowFenw[up_r], W, c, -1)
                    fu(colFenw[c], H, up_r, -1)
            if down_r is not None:
                if grid[down_r][c]:
                    grid[down_r][c] = False
                    removed += 1
                    fu(rowFenw[down_r], W, c, -1)
                    fu(colFenw[c], H, down_r, -1)
            if left_c is not None:
                if grid[r][left_c]:
                    grid[r][left_c] = False
                    removed += 1
                    fu(rowFenw[r], W, left_c, -1)
                    fu(colFenw[left_c], H, r, -1)
            if right_c is not None:
                if grid[r][right_c]:
                    grid[r][right_c] = False
                    removed += 1
                    fu(rowFenw[r], W, right_c, -1)
                    fu(colFenw[right_c], H, r, -1)
    # Final answer: remaining walls = total walls minus those removed.
    ans = H * W - removed
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()