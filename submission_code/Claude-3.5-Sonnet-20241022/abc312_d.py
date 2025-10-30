MOD = 998244353

def solve():
    S = input()
    n = len(S)
    
    # dp[i][j] = number of ways to get balance j after i characters
    dp = [[0]*(n+1) for _ in range(n+1)]
    dp[0][0] = 1
    
    for i in range(n):
        for j in range(n+1):
            if dp[i][j] == 0:
                continue
                
            if S[i] == '(' or S[i] == '?':
                dp[i+1][j+1] = (dp[i+1][j+1] + dp[i][j]) % MOD
                
            if S[i] == ')' or S[i] == '?':
                if j > 0:
                    dp[i+1][j-1] = (dp[i+1][j-1] + dp[i][j]) % MOD
    
    print(dp[n][0])

solve()