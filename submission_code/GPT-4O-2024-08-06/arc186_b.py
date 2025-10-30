# YOUR CODE HERE
def count_valid_permutations(N, A):
    MOD = 998244353
    
    # dp[i] will store the number of valid permutations for the first i elements
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: one way to arrange zero elements
    
    # Prefix sum array to help with range sum queries
    prefix_sum = [0] * (N + 1)
    prefix_sum[0] = dp[0]
    
    for i in range(1, N + 1):
        # Calculate dp[i] using the prefix sum from A[i] to i-1
        dp[i] = (prefix_sum[i - 1] - prefix_sum[A[i - 1]] + MOD) % MOD
        # Update the prefix sum
        prefix_sum[i] = (prefix_sum[i - 1] + dp[i]) % MOD
    
    return dp[N]

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:]))

result = count_valid_permutations(N, A)
print(result)