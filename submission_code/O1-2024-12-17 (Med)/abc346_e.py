def main():
    import sys
    input = sys.stdin.readline
    from collections import Counter

    # Read inputs
    H, W, M = map(int, input().split())
    # row_time[r] = index of the last operation that painted row r+1 (0 if none)
    # row_color[r] = color used in that last operation for row r+1
    # col_time[c] = index of the last operation that painted column c+1 (0 if none)
    # col_color[c] = color used in that last operation for column c+1
    row_time = [0]*H
    row_color = [0]*H
    col_time = [0]*W
    col_color = [0]*W

    # Read and process the M operations
    for i in range(1, M+1):
        T, A, X = map(int, input().split())
        if T == 1:
            # Paint row A with color X
            r = A - 1
            row_time[r] = i
            row_color[r] = X
        else:
            # Paint column A with color X
            c = A - 1
            col_time[c] = i
            col_color[c] = X

    # Count how many rows and columns remain unpainted
    R0 = sum(1 for r in range(H) if row_time[r] == 0)
    C0 = sum(1 for c in range(W) if col_time[c] == 0)

    # We'll accumulate counts of how many cells end up with each color
    color_count = Counter()

    # 1) Cells where row_time=0 and col_time=0 => color=0
    color_count[0] = R0 * C0

    # 2) Cells in unpainted rows but painted columns => color is column's color
    #    and unpainted columns but painted rows => color is row's color
    freq_row_color = Counter()
    freq_col_color = Counter()

    for r in range(H):
        if row_time[r] > 0:
            freq_row_color[row_color[r]] += 1
    for c in range(W):
        if col_time[c] > 0:
            freq_col_color[col_color[c]] += 1

    # Unpainted rows intersect with each painted column
    # => R0 cells per painted column of that color
    for clr, count_cols in freq_col_color.items():
        color_count[clr] += count_cols * R0

    # Unpainted columns intersect with each painted row
    # => C0 cells per painted row of that color
    for clr, count_rows in freq_row_color.items():
        color_count[clr] += count_rows * C0

    # 3) Cells where both row and column are painted.
    #    If row_time[r] = i_r > col_time[c] = i_c, color = row_color[r].
    #    If i_c > i_r, color = col_color[c].
    # We'll gather final row-ops (i_r, color_r) and col-ops (i_c, color_c)
    # only for those rows/columns that actually have a nonzero final paint time.
    row_ops = []
    for r in range(H):
        if row_time[r] > 0:
            row_ops.append((row_time[r], row_color[r]))
    col_ops = []
    for c in range(W):
        if col_time[c] > 0:
            col_ops.append((col_time[c], col_color[c]))

    # Sort each by the operation index
    row_ops.sort(key=lambda x: x[0])  # sort by i_r ascending
    col_ops.sort(key=lambda x: x[0])  # sort by i_c ascending

    # Count pairs where i_r > i_c => color = row_ops[i_r].color
    c_ptr = 0
    lenC = len(col_ops)
    for i_r, clr_r in row_ops:
        while c_ptr < lenC and col_ops[c_ptr][0] < i_r:
            c_ptr += 1
        # c_ptr is the number of columns that have i_c < i_r
        color_count[clr_r] += c_ptr

    # Count pairs where i_c > i_r => color = col_ops[i_c].color
    r_ptr = 0
    lenR = len(row_ops)
    for i_c, clr_c in col_ops:
        while r_ptr < lenR and row_ops[r_ptr][0] < i_c:
            r_ptr += 1
        # r_ptr is the number of rows that have i_r < i_c
        color_count[clr_c] += r_ptr

    # Prepare the output: filter out colors with zero count (must have > 0)
    # Sort by color and print
    final = [(color, cnt) for color, cnt in color_count.items() if cnt > 0]
    final.sort(key=lambda x: x[0])  # ascending by color

    print(len(final))
    for c, cnt in final:
        print(c, cnt)

# Do not forget to call main!
if __name__ == "__main__":
    main()