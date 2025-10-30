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
        row = data[idx]
        idx += 1
        P.append(row)
    
    # Precompute total_black
    total_black = 0
    for i in range(N):
        for j in range(N):
            if P[i][j] == 'B':
                total_black += 1
    
    # Precompute row_sums
    row_sums = []
    for i in range(N):
        count = 0
        for j in range(N):
            if P[i][j] == 'B':
                count += 1
        row_sums.append(count)
    
    # Precompute col_sums
    col_sums = []
    for j in range(N):
        count = 0
        for i in range(N):
            if P[i][j] == 'B':
                count += 1
        col_sums.append(count)
    
    # Precompute row_prefix
    row_prefix = [0] * (N + 1)
    for k in range(1, N + 1):
        row_prefix[k] = row_prefix[k - 1] + row_sums[k - 1]
    
    # Precompute col_prefix
    col_prefix = [0] * (N + 1)
    for k in range(1, N + 1):
        col_prefix[k] = col_prefix[k - 1] + col_sums[k - 1]
    
    # Precompute prefix_sum
    prefix_sum = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i > 0:
                prefix_sum[i][j] += prefix_sum[i - 1][j]
            if j > 0:
                prefix_sum[i][j] += prefix_sum[i][j - 1]
            if i > 0 and j > 0:
                prefix_sum[i][j] -= prefix_sum[i - 1][j - 1]
            if P[i][j] == 'B':
                prefix_sum[i][j] += 1
    
    # Function to compute S(X, Y)
    def S(X, Y):
        if X < 0 or Y < 0:
            return 0
        full_rows = X // N
        full_cols = Y // N
        remainder_rows = X % N
        remainder_cols = Y % N
        result = full_rows * full_cols * total_black
        if remainder_rows > 0:
            result += full_cols * row_prefix[remainder_rows]
        if remainder_cols > 0:
            result += full_rows * col_prefix[remainder_cols]
        if remainder_rows > 0 and remainder_cols > 0:
            result += prefix_sum[remainder_rows - 1][remainder_cols - 1]
        return result
    
    # Process each query
    for _ in range(Q):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        C = int(data[idx])
        idx += 1
        D = int(data[idx])
        idx += 1
        ans = S(C, D) - S(C, B - 1) - S(A - 1, D) + S(A - 1, B - 1)
        print(ans)

if __name__ == '__main__':
    main()