# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = data[2]
    
    MOD = 998244353
    
    # dp[i][j][k] = number of strings of length j with LCS length k with S[:i]
    dp = [[[0] * (N + 1) for _ in range(M + 1)] for _ in range(N + 1)]
    
    # Base case: dp[0][0][0] = 1
    dp[0][0][0] = 1
    
    for i in range(N + 1):
        for j in range(M + 1):
            for k in range(N + 1):
                if dp[i][j][k] == 0:
                    continue
                
                # Option 1: Add a character that doesn't match S[i] (if i < N)
                if j < M:
                    dp[i][j + 1][k] = (dp[i][j + 1][k] + dp[i][j][k] * 25) % MOD
                
                # Option 2: Add a character that matches S[i] (if i < N and k < N)
                if i < N and j < M and k < N:
                    dp[i + 1][j + 1][k + 1] = (dp[i + 1][j + 1][k + 1] + dp[i][j][k]) % MOD
                
                # Option 3: Skip S[i] (if i < N)
                if i < N:
                    dp[i + 1][j][k] = (dp[i + 1][j][k] + dp[i][j][k]) % MOD
    
    # Collect results for each k = 0, 1, ..., N
    result = [dp[N][M][k] for k in range(N + 1)]
    
    print(" ".join(map(str, result)))