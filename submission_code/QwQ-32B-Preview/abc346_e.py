def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    M = int(data[2])
    
    last_row_op = [0] * (H + 1)
    last_col_op = [0] * (W + 1)
    
    idx = 3
    for _ in range(M):
        T = int(data[idx])
        A = int(data[idx + 1])
        X = int(data[idx + 2])
        idx += 3
        if T == 1:
            last_row_op[A] = _
        else:
            last_col_op[A] = _
    
    count_row_ops = [0] * (M + 1)
    for r in range(1, H + 1):
        if last_row_op[r] > 0:
            count_row_ops[last_row_op[r]] += 1
    
    count_col_ops = [0] * (M + 1)
    for c in range(1, W + 1):
        if last_col_op[c] > 0:
            count_col_ops[last_col_op[c]] += 1
    
    R = [0] * (M + 1)
    R[0] = H - sum(count_row_ops[1:])
    for t in range(1, M + 1):
        R[t] = R[t - 1] + count_row_ops[t]
    
    C = [0] * (M + 1)
    C[0] = W - sum(count_col_ops[1:])
    for t in range(1, M + 1):
        C[t] = C[t - 1] + count_col_ops[t]
    
    cells = [0] * (M + 1)
    for t in range(1, M + 1):
        cells[t] = R[t] * C[t] - R[t - 1] * C[t - 1]
    
    cells[0] = R[0] * C[0]
    
    color_counts = {}
    idx = 3
    for t in range(M):
        X = int(data[idx + 2 * t])
        if X in color_counts:
            color_counts[X] += cells[t + 1]
        else:
            color_counts[X] = cells[t + 1]
    
    if 0 in color_counts:
        color_counts[0] += cells[0]
    else:
        color_counts[0] = cells[0]
    
    # Collect and sort colors with non-zero counts
    color_list = [(c, count) for c, count in color_counts.items() if count > 0]
    color_list.sort()
    
    # Output
    print(len(color_list))
    for c, count in color_list:
        print(c, count)

if __name__ == "__main__":
    main()