n, k = map(int, input().split())
a = list(map(int, input().split()))

MOD = 998244353

# Compute prefix sums
prefix = [0] * (n + 1)
for i in range(n):
    prefix[i + 1] = prefix[i] + a[i]

# dp[i] = number of ways to divide A[0:i] such that no subsequence has sum k
dp = [0] * (n + 1)
dp[0] = 1

# sum_dp[p] = sum of dp[j] for all j such that prefix[j] = p
sum_dp = {0: 1}
total_sum_dp = 1

for i in range(1, n + 1):
    bad_prefix = prefix[i] - k
    bad_sum = sum_dp.get(bad_prefix, 0)
    dp[i] = (total_sum_dp - bad_sum) % MOD
    
    sum_dp[prefix[i]] = (sum_dp.get(prefix[i], 0) + dp[i]) % MOD
    total_sum_dp = (total_sum_dp + dp[i]) % MOD

print(dp[n])