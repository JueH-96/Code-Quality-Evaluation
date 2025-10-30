def main():
    import sys
    data = sys.stdin.read().strip().split()
    H, W, N = map(int, data[:3])
    holes = data[3:]

    # Grid of size (H+1)x(W+1), using 1-based indexing
    # Mark 1 if not holed, 0 if holed
    grid = [[1]*(W+1) for _ in range(H+1)]
    idx = 0
    for _ in range(N):
        a = int(holes[idx]); b = int(holes[idx+1])
        grid[a][b] = 0
        idx += 2

    # dp[i][j] will store the size of the largest holeless square
    # whose bottom-right corner is at (i, j)
    dp = [[0]*(W+1) for _ in range(H+1)]

    ans = 0
    for i in range(1, H+1):
        for j in range(1, W+1):
            if grid[i][j] == 1:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                ans += dp[i][j]
    print(ans)

# Do not forget to call main()!
main()