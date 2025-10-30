MOD = 998244353

def count_polish_sequences(N, A):
    # Initialize dp array to store the number of Polish sequences
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: the empty sequence is Polish

    # Initialize a prefix sum array to efficiently compute the sum of dp values
    prefix_sum = [0] * (N + 1)
    prefix_sum[0] = 1

    for i in range(1, N + 1):
        dp[i] = prefix_sum[A[i - 1]]
        prefix_sum[i] = (prefix_sum[i - 1] + dp[i]) % MOD

    return dp[N]

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Calculate the result
result = count_polish_sequences(N, A)

# Print the result
print(result)