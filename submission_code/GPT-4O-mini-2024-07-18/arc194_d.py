def count_distinct_parenthesis_sequences(N, S):
    MOD = 998244353
    
    # dp[i][j] will store the number of distinct valid parenthesis sequences
    dp = [[0] * N for _ in range(N)]
    
    # Base cases
    for i in range(N):
        if i + 1 < N and S[i] == '(' and S[i + 1] == ')':
            dp[i][i + 1] = 1
    
    # Fill the dp table
    for length in range(2, N + 1):  # length of the substring
        for i in range(N - length + 1):
            j = i + length - 1
            if S[i] == '(' and S[j] == ')':
                # Count the valid sequences formed by the substring S[i:j+1]
                dp[i][j] = (dp[i + 1][j - 1] + 1) % MOD  # +1 for the current valid pair
                
                # Now consider all possible partitions
                for k in range(i + 1, j):
                    dp[i][j] = (dp[i][j] + dp[i][k] * dp[k + 1][j]) % MOD
    
    return dp[0][N - 1]

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

# Get the result
result = count_distinct_parenthesis_sequences(N, S)

# Print the result
print(result)