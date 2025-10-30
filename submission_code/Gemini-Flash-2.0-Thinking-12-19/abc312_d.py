def solve():
    s = input()
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    mod = 998244353
    
    for i in range(1, n + 1):
        char = s[i-1]
        for j in range(n + 1):
            if char == '(':
                if j >= 1:
                    dp[i][j] = dp[i-1][j-1]
            elif char == ')':
                if j + 1 <= n:
                    dp[i][j] = dp[i-1][j+1]
            elif char == '?':
                if j >= 1:
                    dp[i][j] = (dp[i][j] + dp[i-1][j-1]) % mod
                if j + 1 <= n:
                    dp[i][j] = (dp[i][j] + dp[i-1][j+1]) % mod
                    
    print(dp[n][0])

if __name__ == '__main__':
    solve()