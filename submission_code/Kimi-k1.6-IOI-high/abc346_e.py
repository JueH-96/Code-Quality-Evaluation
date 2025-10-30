import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    W = int(data[idx+1])
    M = int(data[idx+2])
    idx +=3
    
    row_time = [0] * (H + 1)
    row_color = [0] * (H + 1)
    col_time = [0] * (W + 1)
    col_color = [0] * (W + 1)
    
    for i in range(1, M+1):
        T = int(data[idx])
        A = int(data[idx+1])
        X = int(data[idx+2])
        idx +=3
        if T == 1:
            row_time[A] = i
            row_color[A] = X
        else:
            col_time[A] = i
            col_color[A] = X
    
    painted_rows = []
    for i in range(1, H+1):
        if row_time[i] > 0:
            painted_rows.append((row_time[i], row_color[i]))
    
    painted_cols = []
    for j in range(1, W+1):
        if col_time[j] > 0:
            painted_cols.append((col_time[j], col_color[j]))
    
    R_count = len(painted_rows)
    C_count = len(painted_cols)
    
    color_count = defaultdict(int)
    
    # Unpainted rows and columns
    color_count[0] += (H - R_count) * (W - C_count)
    
    # Painted rows and unpainted columns
    for time, color in painted_rows:
        color_count[color] += (W - C_count)
    
    # Painted columns and unpainted rows
    for time, color in painted_cols:
        color_count[color] += (H - R_count)
    
    # Process rows for intersection
    col_times = [t for t, c in painted_cols]
    col_times.sort()
    for time, color in painted_rows:
        cnt = bisect.bisect_left(col_times, time)
        color_count[color] += cnt
    
    # Process columns for intersection
    row_times = [t for t, c in painted_rows]
    row_times.sort()
    for time, color in painted_cols:
        cnt = bisect.bisect_left(row_times, time)
        color_count[color] += cnt
    
    # Prepare result
    result = []
    for color, count in color_count.items():
        if count > 0:
            result.append((color, count))
    
    result.sort()
    print(len(result))
    for color, count in result:
        print(color, count)

if __name__ == "__main__":
    main()