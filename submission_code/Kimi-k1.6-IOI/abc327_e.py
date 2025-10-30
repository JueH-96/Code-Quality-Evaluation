n = int(input())
p = list(map(int, input().split()))

INF = float('-inf')
dp = [INF] * (n + 1)
dp[0] = 0.0

for score in p:
    # Update dp in reverse order to avoid overwriting issues
    for m in range(n, 0, -1):
        if dp[m-1] != INF:
            candidate = dp[m-1] * 0.9 + score
            if candidate > dp[m]:
                dp[m] = candidate

max_rating = -float('inf')
for m in range(1, n + 1):
    if dp[m] == INF:
        continue
    # Calculate the sum of weights for this m
    weights_sum = 10 * (1 - (0.9 ** m))
    avg = dp[m] / weights_sum
    penalty = 1200 / (m ** 0.5)
    current_rating = avg - penalty
    if current_rating > max_rating:
        max_rating = current_rating

# Print with sufficient precision
print("{0:.12f}".format(max_rating))