# YOUR CODE HERE
def solve():
    N, M = map(int, input().split())
    S = input().strip()
    MOD = 998244353
    
    # dp[i][j][k] = number of strings of length i such that 
    # their LCS with the first j characters of S is exactly k
    dp = [[[0 for _ in range(N+1)] for _ in range(N+1)] for _ in range(M+1)]
    
    # Base cases
    dp[0][0][0] = 1  # Empty string has LCS 0 with empty string
    for j in range(1, N+1):
        dp[0][j][0] = 1  # Empty string has LCS 0 with any string
    
    for i in range(1, M+1):
        dp[i][0][0] = pow(26, i, MOD)  # Any string has LCS 0 with empty string
    
    # Fill the DP table
    for i in range(1, M+1):
        for j in range(1, N+1):
            for k in range(min(i, j) + 1):
                # Append a character that is not S[j-1]
                dp[i][j][k] = (dp[i][j][k] + dp[i-1][j][k] * 25) % MOD
                
                # Append the character S[j-1]
                if k > 0:
                    dp[i][j][k] = (dp[i][j][k] + dp[i-1][j-1][k-1]) % MOD
    
    # The answer for each k is dp[M][N][k]
    answer = [dp[M][N][k] for k in range(N+1)]
    print(*answer)

solve()