# YOUR CODE HERE
def max_experience(N, A):
    # Initialize dp arrays
    dp = [[0, 0] for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        # Let the i-th monster go
        dp[i][0] = max(dp[i][0], dp[i-1][0])
        dp[i][1] = max(dp[i][1], dp[i-1][1])
        
        # Defeat the i-th monster
        # If the previous count of defeated monsters was odd
        dp[i][1] = max(dp[i][1], dp[i-1][0] + 2 * A[i-1])
        # If the previous count of defeated monsters was even
        dp[i][0] = max(dp[i][0], dp[i-1][1] + A[i-1])
    
    # The answer is the maximum experience points possible
    return max(dp[N][0], dp[N][1])

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Calculate and print the result
print(max_experience(N, A))