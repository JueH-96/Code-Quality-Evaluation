import math

n = int(input())
p = list(map(int, input().split()))

max_rating = float('-inf')

# Precompute powers of 0.9 for efficiency
powers = [1.0]
for i in range(1, n):
    powers.append(powers[-1] * 0.9)

for k in range(1, n + 1):
    # DP to find the maximum weighted sum for choosing k contests
    # dp[j] = maximum weighted sum by choosing j contests from those seen so far
    dp = [-float('inf')] * (k + 1)
    dp[0] = 0
    
    for i in range(1, n + 1):
        # Process contest i (1-indexed)
        # Update dp array backwards to avoid using updated values
        for j in range(min(i, k), 0, -1):
            if dp[j-1] != -float('inf'):
                # Weight for the j-th chosen contest when total is k
                weight = powers[k - j]
                dp[j] = max(dp[j], dp[j-1] + weight * p[i-1])
    
    if dp[k] != -float('inf'):
        numerator = dp[k]
        # Calculate denominator: sum of (0.9)^i for i from 0 to k-1
        denominator = sum(powers[i] for i in range(k))
        rating = numerator / denominator - 1200 / math.sqrt(k)
        max_rating = max(max_rating, rating)

print(f"{max_rating:.15f}")