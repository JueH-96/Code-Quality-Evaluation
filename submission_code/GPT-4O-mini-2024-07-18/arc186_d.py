def count_polish_sequences(N, A):
    MOD = 998244353
    
    # Initialize dp array
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: one way to form an empty sequence
    
    # Prefix sums for dp
    prefix_sum = [0] * (N + 1)
    prefix_sum[0] = dp[0]
    
    for i in range(1, N + 1):
        # Calculate dp[i]
        a_i = A[i - 1]
        if a_i > i - 1:
            dp[i] = prefix_sum[i - 1]
        else:
            dp[i] = (prefix_sum[i - 1] - prefix_sum[a_i - 1]) % MOD
        
        # Update prefix sums
        prefix_sum[i] = (prefix_sum[i - 1] + dp[i]) % MOD
    
    return dp[N]

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N + 1]))

# Calculate and print the result
result = count_polish_sequences(N, A)
print(result)