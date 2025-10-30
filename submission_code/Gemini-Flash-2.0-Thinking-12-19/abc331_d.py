def solve():
    n, q = map(int, input().split())
    pattern = [input() for _ in range(n)]
    is_black = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if pattern[i][j] == 'B':
                is_black[i][j] = 1
            else:
                is_black[i][j] = 0
    
    prefix_sum = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            prefix_sum[i][j] = prefix_sum[i-1][j] + prefix_sum[i][j-1] - prefix_sum[i-1][j-1] + is_black[i-1][j-1]
            
    results = []
    for _ in range(q):
        a, b, c, d = map(int, input().split())
        rect_rows = c - a + 1
        rect_cols = d - b + 1
        n_rows_full = rect_rows // n
        n_cols_full = rect_cols // n
        rem_rows = rect_rows % n
        rem_cols = rect_cols % n
        
        black_in_nxn = prefix_sum[n][n]
        black_in_nx_rem_cols = prefix_sum[n][rem_cols] if rem_cols > 0 else 0
        black_in_rem_rows_xn = prefix_sum[rem_rows][n] if rem_rows > 0 else 0
        black_in_rem_rows_x_rem_cols = prefix_sum[rem_rows][rem_cols] if (rem_rows > 0 and rem_cols > 0) else 0
        
        total_black_squares = (n_rows_full * n_cols_full * black_in_nxn) + \
                               (n_rows_full * black_in_nx_rem_cols) + \
                               (n_cols_full * black_in_rem_rows_xn) + \
                               black_in_rem_rows_x_rem_cols
        results.append(total_black_squares)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()