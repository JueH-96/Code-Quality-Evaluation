N, D, P = map(int, input().split())
F = list(map(int, input().split()))
sum_F = sum(F)

# Sort the fares in descending order
F.sort(reverse=True)

# Compute prefix sums
prefix = [0] * (N + 1)
for i in range(1, N + 1):
    prefix[i] = prefix[i - 1] + F[i - 1]

# Calculate the maximum number of batches needed
K_max = (N + D - 1) // D

min_cost = float('inf')

for K in range(K_max + 1):
    passes = K * D
    if passes > N:
        passes = N
    sum_top = prefix[passes]
    cost = K * P + (sum_F - sum_top)
    if cost < min_cost:
        min_cost = cost

print(min_cost)