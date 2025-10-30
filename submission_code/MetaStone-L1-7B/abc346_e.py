import bisect
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    H = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    row_colors = [0] * (H + 1)
    t_row = [0] * (H + 1)
    col_colors = [0] * (W + 1)
    t_col = [0] * (W + 1)
    
    t = 0
    for _ in range(M):
        T = int(data[idx])
        idx += 1
        A = int(data[idx])
        idx += 1
        X = int(data[idx])
        idx += 1
        if T == 1:
            row_colors[A] = X
            t_row[A] = t
        else:
            col_colors[A] = X
            t_col[A] = t
        t += 1
    
    sorted_cols = sorted(t_col[1:W+1])
    sorted_rows = sorted(t_row[1:H+1])
    
    color_counts = defaultdict(int)
    
    for j in range(1, W + 1):
        t = t_col[j]
        count = bisect.bisect_left(sorted_rows, t)
        color = col_colors[j]
        color_counts[color] += count
    
    for i in range(1, H + 1):
        t = t_row[i]
        count = bisect.bisect_left(sorted_cols, t)
        color = row_colors[i]
        color_counts[color] += count
    
    sum_row = sum(bisect.bisect_left(sorted_cols, t) for t in t_row[1:H+1])
    sum_col = sum(bisect.bisect_left(sorted_rows, t) for t in t_col[1:W+1])
    zero_cells = H * W - sum_row - sum_col
    
    color_counts[0] = zero_cells
    
    result = []
    for color in sorted(color_counts):
        if color_counts[color] > 0:
            result.append((color_counts[color], color))
    
    print(len(result))
    for c in result:
        print(c[0], c[1])

if __name__ == '__main__':
    main()