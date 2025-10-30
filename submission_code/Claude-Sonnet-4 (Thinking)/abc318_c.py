N, D, P = map(int, input().split())
F = list(map(int, input().split()))

# Sort fares in descending order to prioritize using passes on expensive days
F.sort(reverse=True)

# Compute prefix sums for efficient range sum queries
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = prefix_sum[i] + F[i]

# Try different numbers of batches
max_batches = (N + D - 1) // D  # This is equivalent to ceil(N / D)

min_cost = prefix_sum[N]  # Cost when k=0 (no batches, pay all fares)

for k in range(1, max_batches + 1):
    passes_available = k * D
    passes_used = min(passes_available, N)
    
    pass_cost = k * P
    fare_cost = prefix_sum[N] - prefix_sum[passes_used]
    
    total_cost = pass_cost + fare_cost
    min_cost = min(min_cost, total_cost)

print(min_cost)