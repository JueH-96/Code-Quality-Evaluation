# YOUR CODE HERE
import sys

def count_permutations(A, B, M):
    N = A * B - 1
    
    # Initialize dp array
    dp = [[[0 for _ in range(B+1)] for _ in range(A+1)] for _ in range(N+1)]
    
    # Base case
    dp[0][0][0] = 1
    
    # Dynamic programming
    for n in range(1, N+1):
        for a in range(A+1):
            for b in range(B+1):
                for k in range(1, n+1):
                    if a > 0:
                        dp[n][a][b] = (dp[n][a][b] + dp[n-1][a-1][b]) % M
                    if b > 0:
                        dp[n][a][b] = (dp[n][a][b] + dp[n-1][a][b-1]) % M
                    dp[n][a][b] = (dp[n][a][b] - dp[n-1][a][b]) % M
    
    # Calculate final result
    result = 0
    for a in range(A):
        for b in range(B):
            result = (result + dp[N][a][b]) % M
    
    return result

# Read input
A, B, M = map(int, input().split())

# Calculate and print result
print(count_permutations(A, B, M))