def main():
    import sys
    import bisect

    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    M = int(data[2])
    
    row_ops = [ (0, 0) for _ in range(H) ]  # (time, color)
    col_ops = [ (0, 0) for _ in range(W) ]  # (time, color)
    
    for i in range(M):
        T = int(data[3 + 3*i])
        A = int(data[4 + 3*i]) - 1  # 0-indexed
        X = int(data[5 + 3*i])
        if T == 1:
            row_ops[A] = (i+1, X)
        else:
            col_ops[A] = (i+1, X)
    
    # Sort column times
    col_times = [col_ops[c][0] for c in range(W)]
    col_times_sorted = sorted(col_times)
    
    # For each row, find count_r: columns with col_time < row_time
    freq = {}
    for r in range(H):
        row_time, row_color = row_ops[r]
        # Number of columns with col_time < row_time
        count_r = bisect.bisect_left(col_times_sorted, row_time)
        if row_color in freq:
            freq[row_color] += count_r
        else:
            freq[row_color] = count_r
    
    # Sort row times
    row_times = [row_ops[r][0] for r in range(H)]
    row_times_sorted = sorted(row_times)
    
    # For each column, find count_c: rows with row_time < col_time
    for c in range(W):
        col_time, col_color = col_ops[c]
        # Number of rows with row_time < col_time
        count_c = bisect.bisect_left(row_times_sorted, col_time)
        if col_color in freq:
            freq[col_color] += count_c
        else:
            freq[col_color] = count_c
    
    # Collect colors with frequency > 0
    colors = [c for c in freq if freq[c] > 0]
    colors.sort()
    
    # Output
    print(len(colors))
    for c in colors:
        print(f"{c} {freq[c]}")

if __name__ == '__main__':
    main()