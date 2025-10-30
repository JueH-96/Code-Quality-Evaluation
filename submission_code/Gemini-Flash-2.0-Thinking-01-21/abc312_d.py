def solve():
    s = input()
    n = len(s)
    mod = 998244353
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(n):
        for j in range(n + 1):
            if dp[i][j] > 0:
                char = s[i]
                if char == '(':
                    if j + 1 <= n:
                        dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]) % mod
                elif char == ')':
                    if j >= 1:
                        dp[i+1][j-1] = (dp[i+1][j-1] + dp[i][j]) % mod
                elif char == '?':
                    if j + 1 <= n:
                        dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]) % mod
                    if j >= 1:
                        dp[i+1][j-1] = (dp[i+1][j-1] + dp[i][j]) % mod
                        
    print(dp[n][0])

if __name__ == '__main__':
    solve()