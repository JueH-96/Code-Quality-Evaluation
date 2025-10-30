def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    H, W, M = map(int, input_data[:3])
    idx = 3
    
    # Store the last repaint time and color for each row/column.
    # time 0 with color 0 means "default" (never repainted).
    row_time  = [0] * H
    row_color = [0] * H
    col_time  = [0] * W
    col_color = [0] * W
    
    # Read the M operations
    # For operation i (1-based), store time = i
    for i in range(1, M+1):
        T_i = int(input_data[idx]); idx += 1
        A_i = int(input_data[idx]); idx += 1
        X_i = int(input_data[idx]); idx += 1
        if T_i == 1:
            # Repaint row A_i with color X_i at time i
            row_time[A_i - 1]  = i
            row_color[A_i - 1] = X_i
        else:
            # Repaint column A_i with color X_i at time i
            col_time[A_i - 1]  = i
            col_color[A_i - 1] = X_i
    
    # Sort row_time and col_time for binary search
    sorted_row_time = sorted(row_time)
    sorted_col_time = sorted(col_time)
    
    # A dictionary to hold final counts for each color
    from collections import defaultdict
    color_count = defaultdict(int)
    
    # For each row r, count how many columns c satisfy col_time[c] <= row_time[r].
    # Those cells get the row's color.
    # Use bisect_right to find how many column-times are <= row_time[r].
    for r in range(H):
        t = row_time[r]
        ccount = bisect.bisect_right(sorted_col_time, t)
        if ccount > 0:
            color_count[row_color[r]] += ccount
    
    # For each column c, count how many rows r satisfy row_time[r] < col_time[c].
    # Those cells get the column's color.
    # Use bisect_left to find how many row-times are < col_time[c].
    for c in range(W):
        t = col_time[c]
        rcount = bisect.bisect_left(sorted_row_time, t)
        if rcount > 0:
            color_count[col_color[c]] += rcount
    
    # Now prepare the output for colors that appear (color_count[color] > 0).
    # Sort colors in ascending order.
    result = []
    for color, cnt in color_count.items():
        if cnt > 0:
            result.append((color, cnt))
    
    result.sort(key=lambda x: x[0])  # sort by color
    
    print(len(result))
    for c, s in result:
        print(c, s)

# do not forget to call main
main()