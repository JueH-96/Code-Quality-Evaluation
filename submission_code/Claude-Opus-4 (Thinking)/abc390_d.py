# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Precompute sums for all subsets
subset_sum = [0] * (1 << n)
for mask in range(1 << n):
    for i in range(n):
        if mask & (1 << i):
            subset_sum[mask] += a[i]

# DP
dp = [set() for _ in range(1 << n)]
dp[0].add(0)

for mask in range(1, 1 << n):
    # Try all non-empty submasks
    submask = mask
    while submask > 0:
        # Partition into submask and mask ^ submask
        complement = mask ^ submask
        for val in dp[complement]:
            dp[mask].add(subset_sum[submask] ^ val)
        submask = (submask - 1) & mask

print(len(dp[(1 << n) - 1]))