def main():
    import sys
    import bisect

    input = sys.stdin.readline
    
    H, W, M = map(int, input().split())
    # Initialize: row_time and col_time (0 means no operation)
    row_time = [0] * H
    row_color = [0] * H
    col_time = [0] * W
    col_color = [0] * W
    
    # Process operations sequentially (time index from 1 to M)
    for t in range(1, M + 1):
        line = input().split()
        if not line:
            continue
        typ = int(line[0])
        a = int(line[1]) - 1  # convert to 0-indexed
        x = int(line[2])
        if typ == 1:
            row_time[a] = t
            row_color[a] = x
        else:
            col_time[a] = t
            col_color[a] = x

    # sort the times to use binary search
    row_times_sorted = sorted(row_time)
    col_times_sorted = sorted(col_time)
    
    from collections import defaultdict
    ans = defaultdict(int)
    
    # For each row that was repainted, count columns where col_time < row_time.
    for r in range(H):
        t_r = row_time[r]
        if t_r == 0:
            continue
        # count columns c with col_time[c] < t_r
        cnt = bisect.bisect_left(col_times_sorted, t_r)
        ans[row_color[r]] += cnt

    # For each column that was repainted, count rows where row_time < col_time.
    for c in range(W):
        t_c = col_time[c]
        if t_c == 0:
            continue
        cnt = bisect.bisect_left(row_times_sorted, t_c)
        ans[col_color[c]] += cnt

    # Account for the cells that were never painted.
    # They remain the initial color, i.e., color 0.
    cnt_rows0 = row_time.count(0)
    cnt_cols0 = col_time.count(0)
    initial_cells = cnt_rows0 * cnt_cols0
    if initial_cells > 0:
        ans[0] += initial_cells

    # Filter the colors with positive cell count and output in sorted order.
    colors = [c for c in ans if ans[c] > 0]
    colors.sort()
    
    output_lines = []
    output_lines.append(str(len(colors)))
    for c in colors:
        output_lines.append(f"{c} {ans[c]}")
    sys.stdout.write("
".join(output_lines))
    
if __name__ == '__main__':
    main()