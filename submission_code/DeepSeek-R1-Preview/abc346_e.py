import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    idx +=1
    W = int(data[idx])
    idx +=1
    M = int(data[idx])
    idx +=1
    
    last_row = [0] * (H + 1)  # 1-based indexing
    row_color = [0] * (H + 1)
    last_col = [0] * (W + 1)
    col_color = [0] * (W + 1)
    
    for step in range(1, M+1):
        T = int(data[idx])
        idx +=1
        A = int(data[idx])
        idx +=1
        X = int(data[idx])
        idx +=1
        if T == 1:
            # Update row A
            last_row[A] = step
            row_color[A] = X
        else:
            # Update column A
            last_col[A] = step
            col_color[A] = X
    
    # Collect column times and sort
    sorted_col_times = sorted(last_col[1:W+1])
    
    # Process each row's contribution
    color_counts = defaultdict(int)
    for r in range(1, H+1):
        time_r = last_row[r]
        if time_r == 0:
            continue
        # Find number of columns with last_col_time < time_r
        count = bisect.bisect_left(sorted_col_times, time_r)
        color = row_color[r]
        color_counts[color] += count
    
    # Collect row times and sort
    sorted_row_times = sorted(last_row[1:H+1])
    
    # Process each column's contribution
    for c in range(1, W+1):
        time_c = last_col[c]
        if time_c == 0:
            continue
        # Find number of rows with last_row_time < time_c
        count = bisect.bisect_left(sorted_row_times, time_c)
        color = col_color[c]
        color_counts[color] += count
    
    # Calculate cells that are 0 because both row and column were never painted
    num_rows_zero = sum(1 for r in range(1, H+1) if last_row[r] == 0)
    num_cols_zero = sum(1 for c in range(1, W+1) if last_col[c] == 0)
    count_0 = num_rows_zero * num_cols_zero
    color_counts[0] += count_0
    
    # Prepare the result, filtering out colors with 0 count
    result = []
    for color in color_counts:
        if color_counts[color] > 0:
            result.append( (color, color_counts[color]) )
    
    # Sort the result by color
    result.sort()
    
    # Output
    print(len(result))
    for color, cnt in result:
        print(color, cnt)

if __name__ == "__main__":
    main()