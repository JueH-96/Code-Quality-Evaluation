def solve():
    n, q = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    prefix_sum = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(n):
            prefix_sum[i+1][j+1] = prefix_sum[i+1][j] + prefix_sum[i][j+1] - prefix_sum[i][j] + (1 if grid[i][j] == 'B' else 0)

    def count_black(r, c):
        full_rows = r // n
        full_cols = c // n
        rem_row = r % n
        rem_col = c % n
        
        total = full_rows * full_cols * prefix_sum[n][n]
        total += full_rows * prefix_sum[n][rem_col]
        total += full_cols * prefix_sum[rem_row][n]
        total += prefix_sum[rem_row][rem_col]
        return total

    for _ in range(q):
        a, b, c, d = map(int, input().split())
        ans = count_black(c + 1, d + 1) - count_black(a, d + 1) - count_black(c + 1, b) + count_black(a, b)
        print(ans)

solve()