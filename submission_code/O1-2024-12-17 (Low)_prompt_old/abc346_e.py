def solve():
    import sys
    import bisect
    
    input_data = sys.stdin.read().strip().split()
    H, W, M = map(int, input_data[:3])
    idx = 3
    
    # We will keep track of the last operation index (1-based) that repainted each row/column,
    # along with the color used. If no operation touched a row/column, its time remains 0.
    row_time = [0]*H
    row_color = [0]*H
    col_time = [0]*W
    col_color = [0]*W
    
    # Read operations
    # T_i A_i X_i
    # If T_i=1 => row A_i (1-based), color X_i
    # If T_i=2 => column A_i (1-based), color X_i
    for i in range(1, M+1):
        T = int(input_data[idx]); idx+=1
        A = int(input_data[idx]); idx+=1
        X = int(input_data[idx]); idx+=1
        if T == 1:
            # repaint row
            row_time[A-1] = i
            row_color[A-1] = X
        else:
            # repaint column
            col_time[A-1] = i
            col_color[A-1] = X
    
    # We now have, for each row r, (row_time[r], row_color[r]),
    # and for each column c, (col_time[c], col_color[c]).
    #
    # Final color of cell (r, c) is determined by comparing row_time[r] and col_time[c]:
    # - If row_time[r] > col_time[c], cell is row_color[r].
    # - If row_time[r] < col_time[c], cell is col_color[c].
    # - If both are 0, color is 0.
    # - (They can't be equal except both 0, because each operation index is unique.)
    #
    # We want to count how many cells each color has in the end.
    #
    # Let row_time_all = sorted list of all row_time[r].
    # For each row r with time t_r>0, the number of columns c with col_time[c] < t_r
    # is the number of cells in that row that end up with row_color[r]. We add that to that color's count.
    #
    # Similarly, for each column c with time t_c>0, the number of rows r with row_time[r] < t_c
    # is the number of cells in that column that end up with col_color[c]. We add that to that color's count.
    #
    # Finally, the cells where row_time[r] = col_time[c] = 0 remain color 0.
    #
    # Implementation detail: we must include zeros in the sorted arrays so that
    # bisect_left(...) also counts columns/rows with time=0 properly.
    
    # Build sorted arrays
    # row_time_all sorted of length H
    # col_time_all sorted of length W
    row_time_all = sorted(row_time)
    col_time_all = sorted(col_time)
    
    # We'll need to count how many times each color appears.
    from collections import defaultdict
    color_count = defaultdict(int)
    
    # For each row r:
    # let t_r = row_time[r], c_r = row_color[r].
    # We add j = number of columns c whose col_time[c] < t_r to color_count[c_r].
    # We find j via binary search in col_time_all.
    # Note: if t_r=0, we add nothing because j=bisect_left(col_time_all,0)=0 => no overshadow from row side.
    for r in range(H):
        t_r = row_time[r]
        if t_r > 0:
            c_r = row_color[r]
            j = bisect.bisect_left(col_time_all, t_r)  # number of col_time < t_r
            if j > 0:
                color_count[c_r] += j
    
    # For each column c:
    # let t_c = col_time[c], c_c = col_color[c].
    # The number of rows r whose row_time[r]<t_c is i = bisect_left(row_time_all, t_c).
    # Add i to color_count[c_c].
    for c in range(W):
        t_c = col_time[c]
        if t_c > 0:
            c_c = col_color[c]
            i = bisect.bisect_left(row_time_all, t_c)  # number of row_time < t_c
            if i > 0:
                color_count[c_c] += i
    
    # Now, the cells where row_time[r]=0 and col_time[c]=0 remain color 0.
    # Count how many rows have time=0, how many columns have time=0
    rows_zero = sum(1 for t in row_time if t==0)
    cols_zero = sum(1 for t in col_time if t==0)
    if rows_zero > 0 and cols_zero > 0:
        color_count[0] += rows_zero * cols_zero
    
    # Now we have the counts for each color. We want to output the number of distinct colors >0
    # and then each color and count in ascending order.
    # Some colors might have count=0 (e.g. if they were overwritten completely),
    # so we only keep those with count>0.
    
    # Filter out zero-count
    final_colors = [(c, color_count[c]) for c in color_count if color_count[c] > 0]
    final_colors.sort(key=lambda x: x[0])  # sort by color ascending
    
    print(len(final_colors))
    for c, cnt in final_colors:
        print(c, cnt)