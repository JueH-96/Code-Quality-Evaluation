def main():
    import sys
    from bisect import bisect_left

    data = sys.stdin.read().strip().split()
    H = int(data[0])
    W = int(data[1])
    M = int(data[2])

    # For each row/column, store the index of the last operation that repainted it
    # (rowTime[r] or colTime[c]) and the color used (rowColor[r] or colColor[c]).
    rowTime = [0]*H
    rowColor = [0]*H
    colTime = [0]*W
    colColor = [0]*W

    idx = 3
    for i in range(1, M+1):
        t = int(data[idx])
        a = int(data[idx+1]) - 1  # make 0-based
        x = int(data[idx+2])
        idx += 3

        if t == 1:
            # Repaint row a with color x
            rowTime[a] = i
            rowColor[a] = x
        else:
            # Repaint column a with color x
            colTime[a] = i
            colColor[a] = x

    # We will count how many cells end in each color using a sweep approach:
    # 1) Rows overshadow: each updated row "wins" over columns whose last-update-time is strictly smaller.
    # 2) Columns overshadow: each updated column "wins" over rows whose last-update-time is strictly smaller.
    # 3) Cells where both row and column are 0 (never updated) remain color 0.

    # Prepare to count contributions from rows.
    # Collect (time, color) for every row that has time>0 (meaning it was updated).
    updatedRows = []
    for r in range(H):
        if rowTime[r] > 0:
            updatedRows.append((rowTime[r], rowColor[r]))
    updatedRows.sort(key=lambda x: x[0])  # sort by time

    # Sort all column times so we can quickly find how many columns have colTime < rowTime[r].
    colTimesSorted = sorted(colTime)

    colorCount = {}
    def add_color(c, v):
        if v > 0:
            colorCount[c] = colorCount.get(c, 0) + v

    # Sweep through updatedRows in ascending order of rowTime; track how many columns are < that time
    pc = 0
    overshadowed = 0
    Wlen = W
    for (t_r, c_r) in updatedRows:
        # Advance pc while colTimesSorted[pc] < t_r
        while pc < Wlen and colTimesSorted[pc] < t_r:
            pc += 1
            overshadowed += 1
        # All those columns with colTime < t_r are "overshadowed" by this row's color
        add_color(c_r, overshadowed)

    # Now count column overshadows: each updated column "wins" over rows with rowTime < colTime[c].
    updatedCols = []
    for c in range(W):
        if colTime[c] > 0:
            updatedCols.append((colTime[c], colColor[c]))
    updatedCols.sort(key=lambda x: x[0])

    rowTimesSorted = sorted(rowTime)
    pr = 0
    overshadowedRows = 0
    Hlen = H
    for (t_c, c_c) in updatedCols:
        while pr < Hlen and rowTimesSorted[pr] < t_c:
            pr += 1
            overshadowedRows += 1
        add_color(c_c, overshadowedRows)

    # Finally, cells where rowTime=0 and colTime=0 remain color 0
    unupdatedRows = sum(1 for r in range(H) if rowTime[r] == 0)
    unupdatedCols = sum(1 for c in range(W) if colTime[c] == 0)
    add_color(0, unupdatedRows * unupdatedCols)

    # Prepare output: only colors with positive counts
    items = [(c, colorCount[c]) for c in colorCount if colorCount[c] > 0]
    items.sort(key=lambda x: x[0])  # sort by color

    print(len(items))
    for c, cnt in items:
        print(c, cnt)

# Do not forget to call main().
main()