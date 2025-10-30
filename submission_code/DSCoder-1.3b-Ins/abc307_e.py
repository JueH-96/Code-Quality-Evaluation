def main():
    N, M = map(int, input().split())
    mod = 998244353
    dp = [[0]*M for _ in range(N+1)]
    dp[1][0] = 1
    for i in range(2, N+1):
        for j in range(M):
            dp[i][j] = (dp[i-1][j] + dp[i-1][(j+1)%M]) % mod
    print(dp[N][0])

if __name__ == "__main__":
    main()