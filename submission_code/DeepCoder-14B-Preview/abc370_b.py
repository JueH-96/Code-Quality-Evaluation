MOD = 998244353

n, k = map(int, input().split())
a = list(map(int, input().split()))

# Compute prefix sums
s = [0] * (n + 1)
for i in range(1, n + 1):
    s[i] = s[i-1] + a[i-1]

dp = [0] * (n + 1)
dp[0] = 1

total_sum = dp[0]  # This holds the sum of dp[0..i-1] for the current i

sum_counts = {s[0]: dp[0] % MOD}

for i in range(1, n + 1):
    current_s = s[i]
    target = current_s - k
    sum_invalid = sum_counts.get(target, 0)
    
    dp_i = (total_sum - sum_invalid) % MOD
    dp[i] = dp_i
    
    # Update sum_counts and total_sum
    sum_counts[current_s] = (sum_counts.get(current_s, 0) + dp_i) % MOD
    total_sum = (total_sum + dp_i) % MOD

print(dp[n] % MOD)