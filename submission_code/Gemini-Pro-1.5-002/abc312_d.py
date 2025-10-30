# YOUR CODE HERE
MOD = 998244353

def solve():
    s = input()
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(n):
        for j in range(n):
            if s[i] != '(':
                dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]) % MOD
            if s[i] != ')':
                if j > 0:
                    dp[i+1][j-1] = (dp[i+1][j-1] + dp[i][j]) % MOD
    
    print(dp[n][0])

solve()