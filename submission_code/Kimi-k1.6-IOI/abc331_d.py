def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    P = []
    for _ in range(N):
        P.append(data[idx])
        idx += 1
    
    # Precompute prefix sum
    pre_sum = [[0]*(N+1) for _ in range(N+1)]
    for i in range(N):
        for j in range(N):
            pre_sum[i+1][j+1] = pre_sum[i][j+1] + pre_sum[i+1][j] - pre_sum[i][j] + (P[i][j] == 'B')
    
    # Precompute row_B and col_B
    row_B = [pre_sum[i+1][N] - pre_sum[i][N] for i in range(N)]
    col_B = [pre_sum[N][j+1] - pre_sum[N][j] for j in range(N)]
    
    # Precompute row_prefix and col_prefix
    row_prefix = [0] * (N + 1)
    for i in range(N):
        row_prefix[i+1] = row_prefix[i] + row_B[i]
    
    col_prefix = [0] * (N + 1)
    for j in range(N):
        col_prefix[j+1] = col_prefix[j] + col_B[j]
    
    total_B = pre_sum[N][N]
    
    results = []
    for _ in range(Q):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        D = int(data[idx])
        idx += 1
        
        H = C - A + 1
        W = D - B + 1
        
        H_full, H_rem = divmod(H, N)
        W_full, W_rem = divmod(W, N)
        
        sum_full = H_full * W_full * total_B
        
        sum_partial_rows = 0
        if H_rem > 0:
            start_row = A % N
            end_row = (A + H_rem - 1) % N
            if start_row <= end_row:
                row_intervals = [(start_row, end_row)]
            else:
                row_intervals = [(start_row, N-1), (0, end_row)]
            sum_rows = 0
            for r1, r2 in row_intervals:
                sum_rows += row_prefix[r2+1] - row_prefix[r1]
            sum_partial_rows = sum_rows * W_full
        
        sum_partial_cols = 0
        if W_rem > 0:
            start_col = B % N
            end_col = (B + W_rem - 1) % N
            if start_col <= end_col:
                col_intervals = [(start_col, end_col)]
            else:
                col_intervals = [(start_col, N-1), (0, end_col)]
            sum_cols = 0
            for c1, c2 in col_intervals:
                sum_cols += col_prefix[c2+1] - col_prefix[c1]
            sum_partial_cols = sum_cols * H_full
        
        sum_partial = 0
        if H_rem > 0 and W_rem > 0:
            start_row = A % N
            end_row = (A + H_rem - 1) % N
            if start_row <= end_row:
                row_intervals = [(start_row, end_row)]
            else:
                row_intervals = [(start_row, N-1), (0, end_row)]
            
            start_col = B % N
            end_col = (B + W_rem - 1) % N
            if start_col <= end_col:
                col_intervals = [(start_col, end_col)]
            else:
                col_intervals = [(start_col, N-1), (0, end_col)]
            
            sum_partial = 0
            for r1, r2 in row_intervals:
                for c1, c2 in col_intervals:
                    sum_partial += pre_sum[r2+1][c2+1] - pre_sum[r1][c2+1] - pre_sum[r2+1][c1] + pre_sum[r1][c1]
        
        total = sum_full + sum_partial_rows + sum_partial_cols + sum_partial
        results.append(total)
    
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == '__main__':
    main()