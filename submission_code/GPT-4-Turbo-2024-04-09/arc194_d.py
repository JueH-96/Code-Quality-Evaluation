def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    MOD = 998244353
    
    # dp[i][j] will be the number of distinct valid parenthesis sequences from S[i:j+1]
    dp = [[0] * N for _ in range(N)]
    
    # Base case: single character sequences
    for i in range(N):
        if S[i] == '(':  # Only '(' can start a valid sequence
            dp[i][i] = 1
    
    # Fill dp for longer substrings
    for length in range(2, N + 1):  # length of the substring
        for i in range(N - length + 1):
            j = i + length - 1
            if S[i] == '(' and S[j] == ')':
                # Case 1: Entirely wrapping a valid sequence
                if i + 1 <= j - 1:
                    dp[i][j] = dp[i + 1][j - 1]
                else:
                    dp[i][j] = 1  # Empty inside, valid "()"
                
                # Case 2: Concatenation of two valid sequences
                for k in range(i, j):
                    dp[i][j] += dp[i][k] * dp[k + 1][j]
                    dp[i][j] %= MOD
    
    # The result for the whole string S[0:N] is in dp[0][N-1]
    print(dp[0][N-1])