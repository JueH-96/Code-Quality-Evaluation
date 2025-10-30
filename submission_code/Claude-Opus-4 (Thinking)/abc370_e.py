# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

MOD = 998244353

# Compute prefix sums
prefix = [0] * (N + 1)
for i in range(N):
    prefix[i + 1] = prefix[i] + A[i]

# dp[i] = number of valid divisions of A[0:i]
dp = [0] * (N + 1)
dp[0] = 1  # Base case

# Dictionary to store sum of dp values for each prefix sum
prefix_dp_sum = {}
prefix_dp_sum[0] = 1
total_sum = 1

for i in range(1, N + 1):
    # We want to sum dp[j] for all j < i where sum(A[j:i]) != K
    # sum(A[j:i]) = prefix[i] - prefix[j]
    # So we exclude j where prefix[j] = prefix[i] - K
    target = prefix[i] - K
    excluded_sum = prefix_dp_sum.get(target, 0)
    dp[i] = (total_sum - excluded_sum + MOD) % MOD
    
    # Update the dictionary and total sum
    if prefix[i] not in prefix_dp_sum:
        prefix_dp_sum[prefix[i]] = 0
    prefix_dp_sum[prefix[i]] = (prefix_dp_sum[prefix[i]] + dp[i]) % MOD
    total_sum = (total_sum + dp[i]) % MOD

print(dp[N])