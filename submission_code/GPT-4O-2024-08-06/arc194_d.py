# YOUR CODE HERE
def count_distinct_sequences(N, S):
    MOD = 998244353

    # dp[i][j] will store the number of distinct valid sequences that can be formed from S[i:j+1]
    dp = [[0] * N for _ in range(N)]
    
    # Base case: single pair ()
    for i in range(N - 1):
        if S[i] == '(' and S[i + 1] == ')':
            dp[i][i + 1] = 1
    
    # Fill dp table
    for length in range(2, N):
        for i in range(N - length):
            j = i + length
            if S[i] == '(' and S[j] == ')':
                # Case 1: The whole substring S[i:j+1] is a valid sequence
                dp[i][j] = (dp[i][j] + dp[i + 1][j - 1]) % MOD
            
            # Case 2: Split into two parts
            for k in range(i, j):
                dp[i][j] = (dp[i][j] + dp[i][k] * dp[k + 1][j]) % MOD
    
    # The answer is the number of distinct sequences that can be formed from the whole string
    return dp[0][N - 1]

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

result = count_distinct_sequences(N, S)
print(result)