MOD = 998244353

N, K = map(int, input().split())
A = list(map(int, input().split()))

# Precompute prefix sums
prefix = [0] * (N + 1)
for i in range(N):
    prefix[i + 1] = prefix[i] + A[i]

# dp[i] = number of valid ways to divide A[0:i]
dp = [0] * (N + 1)
dp[0] = 1

for i in range(1, N + 1):
    for j in range(i):
        # Sum of subsequence from j to i-1
        subseq_sum = prefix[i] - prefix[j]
        
        if subseq_sum != K:
            dp[i] = (dp[i] + dp[j]) % MOD

print(dp[N])