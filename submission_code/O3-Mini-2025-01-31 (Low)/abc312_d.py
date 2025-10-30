def main():
    import sys
    sys.setrecursionlimit(10000)
    MOD = 998244353
    s = sys.stdin.readline().strip()
    n = len(s)

    # dp[i][j] = number of ways to form a valid prefix until index i
    # with a balance of j (i.e. j more '(' than ')')
    dp = [[0]*(n+2) for _ in range(n+1)]
    dp[0][0] = 1

    for i in range(n):
        char = s[i]
        for j in range(n+1):
            if dp[i][j] == 0:
                continue
            if char == '(' or char == '?':
                dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]) % MOD
            if (char == ')' or char == '?') and j > 0:
                dp[i+1][j-1] = (dp[i+1][j-1] + dp[i][j]) % MOD

    print(dp[n][0] % MOD)

if __name__ == "__main__":
    main()