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
    
    row_time = [0] * H
    row_color = [0] * H
    col_time = [0] * W
    col_color = [0] * W
    
    for i in range(M):
        T = int(data[idx])
        idx +=1
        A = int(data[idx])
        idx +=1
        X = int(data[idx])
        idx +=1
        if T == 1:
            idx_row = A -1
            row_time[idx_row] = i+1
            row_color[idx_row] = X
        else:
            idx_col = A -1
            col_time[idx_col] = i+1
            col_color[idx_col] = X
    
    sorted_cols = sorted(col_time)
    sorted_rows = sorted(row_time)
    
    freq = defaultdict(int)
    
    for r in range(H):
        t = row_time[r]
        if t ==0:
            continue
        cnt = bisect.bisect_left(sorted_cols, t)
        freq[row_color[r]] += cnt
    
    for c in range(W):
        t = col_time[c]
        if t ==0:
            continue
        cnt = bisect.bisect_left(sorted_rows, t)
        freq[col_color[c]] += cnt
    
    rows_zero = row_time.count(0)
    cols_zero = col_time.count(0)
    color_zero = rows_zero * cols_zero
    freq[0] += color_zero
    
    result = []
    for color, count in freq.items():
        if count >0:
            result.append( (color, count) )
    
    result.sort()
    
    print(len(result))
    for color, count in result:
        print(color, count)
        
if __name__ == '__main__':
    main()